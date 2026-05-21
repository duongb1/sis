import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


LABEL_TO_ID = {"khong": 0, "co": 1}
ID_TO_LABEL = {0: "khong", 1: "co"}


# Large-text teacher -> paired-text student with CE + KD.
# Paired text layout:
# /kaggle/input/datasets/duongb/cthsis/images/{train,val,test}/{co,khong}/{patient_id}/{patient_id}.txt
IMAGE_ROOT = "/kaggle/input/datasets/duongb/cthsis/images"
TEACHER_MODEL_DIR = "/kaggle/working/text_phobert_classifier/best_auc_phobert"
OUTPUT_DIR = "/kaggle/working/paired_text_kd_from_large_text_teacher"

STUDENT_INIT = "vinai/phobert-base"
SPLITS = ["train", "val", "test"]
LABELS = ["co", "khong"]

MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 8
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
THRESHOLD = 0.5
SEED = 42
NUM_WORKERS = 0
FORCE_CPU = False
USE_MULTI_GPU = True
GRADIENT_ACCUMULATION_STEPS = 1

ALPHA_KD = 0.1
KD_LOSS_TYPE = "binary"  # "binary" or "kl"
TEMPERATURE = 2.0
CE_WARMUP_EPOCHS = 0
BEST_MODEL_METRIC = "auc"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    print(f"Using CUDA devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    return torch.device("cuda")


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def wrap_model_for_device(model, device):
    model = model.to(device)
    if device.type == "cuda" and USE_MULTI_GPU and torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        print(f"Using DataParallel on GPU ids: {device_ids}")
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model


def read_text_file(path):
    for encoding in ("utf-8", "utf-8-sig", "cp1258", "latin-1"):
        try:
            return Path(path).read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def resolve_image_root(image_root):
    image_root = Path(image_root)
    if (image_root / "train").exists():
        return image_root
    if (image_root / "lan1-full" / "train").exists():
        return image_root / "lan1-full"
    raise FileNotFoundError(f"Cannot find split folders under {image_root}.")


def resolve_max_length(model, requested_max_length):
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is None:
        return requested_max_length
    return min(requested_max_length, max_positions - 2)


def collect_paired_text_records(image_root):
    image_root = resolve_image_root(image_root)
    records = []
    missing_text = []

    for split in SPLITS:
        for label in LABELS:
            label_dir = image_root / split / label
            if not label_dir.exists():
                continue

            for patient_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
                text_path = patient_dir / f"{patient_dir.name}.txt"
                patient_id = f"{split}/{label}/{patient_dir.name}"
                if not text_path.exists():
                    missing_text.append(patient_id)
                    continue

                text = read_text_file(text_path).strip()
                if not text:
                    missing_text.append(patient_id)
                    continue

                records.append(
                    {
                        "patient_id": patient_id,
                        "split": split,
                        "label": LABEL_TO_ID[label],
                        "label_name": label,
                        "text_path": str(text_path),
                        "text": text,
                    }
                )

    return records, missing_text, image_root


def split_records(records, split):
    return [record for record in records if record["split"] == split]


class TextDataset(Dataset):
    def __init__(self, records, tokenizer, max_length, teacher_logits=None):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.teacher_logits = teacher_logits

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        encoded = self.tokenizer(
            record["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(record["label"], dtype=torch.long)
        if self.teacher_logits is not None:
            item["teacher_logits"] = torch.tensor(self.teacher_logits[record["patient_id"]], dtype=torch.float32)
        item["patient_id"] = record["patient_id"]
        return item


def batch_to_device(batch, device):
    patient_ids = batch.pop("patient_id")
    inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    return inputs, patient_ids


def round_float(value, digits=3):
    return round(float(value), digits)


def rounded_metrics(metrics, digits=3):
    rounded = {}
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            rounded[key] = round_float(value, digits)
        else:
            rounded[key] = value
    return rounded


@torch.no_grad()
def compute_teacher_logits(records, teacher, tokenizer, max_length, device):
    loader = DataLoader(
        TextDataset(records, tokenizer, max_length),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    teacher.eval()
    logits_by_patient = {}
    labels_all = []
    probs_all = []

    for batch in tqdm(loader, desc="Computing large-text teacher logits"):
        inputs, patient_ids = batch_to_device(batch, device)
        labels = inputs.pop("labels")
        outputs = teacher(**inputs)
        logits = outputs.logits.detach().cpu()
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()

        for patient_id, logit in zip(patient_ids, logits):
            logits_by_patient[patient_id] = logit.tolist()
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        probs_all.extend(probs.tolist())

    labels = np.array(labels_all, dtype=np.int64)
    probs = np.array(probs_all, dtype=np.float32)
    preds = (probs >= THRESHOLD).astype(np.int64)
    stats = {
        "teacher_accuracy": float(accuracy_score(labels, preds)),
        "teacher_f1": float(f1_score(labels, preds, zero_division=0)),
        "teacher_auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else float("nan"),
        "teacher_num_patients": int(len(labels)),
        "teacher_num_co": int((labels == 1).sum()),
        "teacher_num_khong": int((labels == 0).sum()),
    }
    return logits_by_patient, stats


def kd_loss(student_logits, teacher_logits, labels, epoch):
    ce_per_sample = F.cross_entropy(student_logits, labels, reduction="none")
    if epoch <= CE_WARMUP_EPOCHS or ALPHA_KD <= 0:
        zero = torch.zeros((), device=student_logits.device)
        return ce_per_sample.mean(), ce_per_sample.mean().detach(), zero

    if KD_LOSS_TYPE == "binary":
        teacher_p_co = F.softmax(teacher_logits, dim=-1)[:, 1].detach()
        student_logit_co = student_logits[:, 1] - student_logits[:, 0]
        kd_per_sample = F.binary_cross_entropy_with_logits(student_logit_co, teacher_p_co, reduction="none")
    elif KD_LOSS_TYPE == "kl":
        kd_per_sample = F.kl_div(
            F.log_softmax(student_logits / TEMPERATURE, dim=-1),
            F.softmax(teacher_logits / TEMPERATURE, dim=-1),
            reduction="none",
        ).sum(dim=-1) * (TEMPERATURE ** 2)
    else:
        raise ValueError(f"Unsupported KD_LOSS_TYPE: {KD_LOSS_TYPE}")

    loss_per_sample = (1.0 - ALPHA_KD) * ce_per_sample + ALPHA_KD * kd_per_sample
    return loss_per_sample.mean(), ce_per_sample.mean().detach(), kd_per_sample.mean().detach()


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_count = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="Training paired CE+KD", leave=False), start=1):
        inputs, _ = batch_to_device(batch, device)
        labels = inputs["labels"]
        teacher_logits = inputs.pop("teacher_logits")

        use_amp = device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(**inputs)
            loss, ce, kd = kd_loss(outputs.logits, teacher_logits, labels, epoch)
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()
        if step % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = labels.size(0)
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS * batch_size
        total_ce += ce.item() * batch_size
        total_kd += kd.item() * batch_size
        total_count += batch_size

    denom = max(total_count, 1)
    return float(total_loss / denom), float(total_ce / denom), float(total_kd / denom)


@torch.no_grad()
def evaluate(model, loader, device, threshold=THRESHOLD):
    model.eval()
    total_loss = 0.0
    total_count = 0
    patient_ids = []
    labels_all = []
    probs_all = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        inputs, batch_patient_ids = batch_to_device(batch, device)
        inputs.pop("teacher_logits", None)
        labels = inputs["labels"]
        outputs = model(**inputs)
        loss = outputs.loss.mean()
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1]

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        patient_ids.extend(batch_patient_ids)
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        probs_all.extend(probs.detach().cpu().numpy().tolist())

    labels = np.array(labels_all, dtype=np.int64)
    probs = np.array(probs_all, dtype=np.float32)
    preds = (probs >= threshold).astype(np.int64)
    metrics = {
        "loss": float(total_loss / max(total_count, 1)),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else float("nan"),
        "num_patients": int(len(labels)),
        "num_co": int((labels == 1).sum()),
        "num_khong": int((labels == 0).sum()),
    }
    return metrics, patient_ids, labels, probs, preds


def save_predictions(path, patient_ids, labels, probs, preds):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["patient_id", "true_label", "true_name", "prob_co", "pred_label", "pred_name"],
        )
        writer.writeheader()
        for patient_id, label, prob, pred in zip(patient_ids, labels, probs, preds):
            writer.writerow(
                {
                    "patient_id": patient_id,
                    "true_label": int(label),
                    "true_name": ID_TO_LABEL[int(label)],
                    "prob_co": round_float(prob),
                    "pred_label": int(pred),
                    "pred_name": ID_TO_LABEL[int(pred)],
                }
            )


def save_dataset_records(path, records):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "split", "label", "label_name", "text_path"])
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in ["patient_id", "split", "label", "label_name", "text_path"]})


def save_model(model, tokenizer, path, epoch, val_metrics, max_length):
    path.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(path)
    tokenizer.save_pretrained(path)
    with open(path / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "epoch": epoch,
                "best_model_metric": BEST_MODEL_METRIC,
                "val_metrics": rounded_metrics(val_metrics),
                "student_init": STUDENT_INIT,
                "teacher_model_dir": TEACHER_MODEL_DIR,
                "max_length": max_length,
                "alpha_kd": ALPHA_KD,
                "kd_loss_type": KD_LOSS_TYPE,
                "temperature": TEMPERATURE,
                "temperature_used": KD_LOSS_TYPE == "kl",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    seed_everything(SEED)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(FORCE_CPU)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    records, missing_text, image_root = collect_paired_text_records(IMAGE_ROOT)
    if missing_text:
        (output_dir / "missing_text_patients.txt").write_text("\n".join(missing_text), encoding="utf-8")

    train_records = split_records(records, "train")
    val_records = split_records(records, "val")
    test_records = split_records(records, "test")
    if not train_records or not val_records:
        raise RuntimeError("Need non-empty train and val records.")

    print(f"Image root: {image_root.resolve()}")
    print(f"Paired text patients: {len(records)}")
    print(f"Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}")
    print(f"Missing/empty text patients: {len(missing_text)}")

    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_DIR, use_fast=False)
    teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_MODEL_DIR)
    teacher_max_length = resolve_max_length(teacher, MAX_LENGTH)
    teacher = wrap_model_for_device(teacher, device)
    teacher_logits, teacher_stats = compute_teacher_logits(records, teacher, teacher_tokenizer, teacher_max_length, device)
    print(f"Teacher on paired data: {rounded_metrics(teacher_stats)}")
    with open(output_dir / "teacher_stats.json", "w", encoding="utf-8") as f:
        json.dump(rounded_metrics(teacher_stats), f, ensure_ascii=False, indent=2)
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_INIT, use_fast=False)
    student = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_INIT,
        num_labels=2,
        id2label={0: "khong", 1: "co"},
        label2id={"khong": 0, "co": 1},
    )
    student_max_length = resolve_max_length(student, MAX_LENGTH)
    if student_max_length != MAX_LENGTH:
        print(f"Requested MAX_LENGTH={MAX_LENGTH}, using {student_max_length}.")
    student = wrap_model_for_device(student, device)

    train_loader = DataLoader(
        TextDataset(train_records, student_tokenizer, student_max_length, teacher_logits),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TextDataset(val_records, student_tokenizer, student_max_length, teacher_logits),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        TextDataset(test_records, student_tokenizer, student_max_length, teacher_logits),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    ) if test_records else None

    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    updates_per_epoch = int(np.ceil(len(train_loader) / max(GRADIENT_ACCUMULATION_STEPS, 1)))
    total_steps = max(1, updates_per_epoch * EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_score = -1.0
    best_dir = output_dir / "best_auc_paired_text_kd"
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_ce, train_kd = train_one_epoch(student, train_loader, optimizer, scheduler, scaler, device, epoch)
        val_metrics, val_ids, val_labels, val_probs, val_preds = evaluate(student, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": round_float(train_loss),
            "train_ce": round_float(train_ce),
            "train_kd": round_float(train_kd),
            "val_loss": round_float(val_metrics["loss"]),
            "val_acc": round_float(val_metrics["accuracy"]),
            "val_f1": round_float(val_metrics["f1"]),
            "val_auc": round_float(val_metrics["auc"]),
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{EPOCHS} | train_loss={train_loss:.3f} | ce={train_ce:.3f} | kd={train_kd:.3f} | "
            f"val_loss={val_metrics['loss']:.3f} | val_acc={val_metrics['accuracy']:.3f} | "
            f"val_f1={val_metrics['f1']:.3f} | val_auc={val_metrics['auc']:.3f}"
        )

        score = val_metrics[BEST_MODEL_METRIC]
        if score > best_score:
            best_score = score
            save_model(student, student_tokenizer, best_dir, epoch, val_metrics, student_max_length)
            save_predictions(output_dir / "val_predictions_best_auc.csv", val_ids, val_labels, val_probs, val_preds)

    with open(output_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    save_dataset_records(output_dir / "dataset_records.csv", records)

    eval_model = AutoModelForSequenceClassification.from_pretrained(best_dir)
    eval_model = wrap_model_for_device(eval_model, device)

    all_metrics = {}
    for split, loader in [("val", val_loader), ("test", test_loader)]:
        if loader is None:
            continue
        metrics, ids, labels, probs, preds = evaluate(eval_model, loader, device, THRESHOLD)
        all_metrics[split] = rounded_metrics(metrics)
        save_predictions(output_dir / f"{split}_predictions_best_auc.csv", ids, labels, probs, preds)
        print(f"{split}: {rounded_metrics(metrics)}")
        print(confusion_matrix(labels, preds, labels=[0, 1]))

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "image_root": str(image_root),
                "teacher_model_dir": TEACHER_MODEL_DIR,
                "student_init": STUDENT_INIT,
                "requested_max_length": MAX_LENGTH,
                "max_length": student_max_length,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "warmup_ratio": WARMUP_RATIO,
                "threshold": THRESHOLD,
                "best_model_metric": BEST_MODEL_METRIC,
                "alpha_kd": ALPHA_KD,
                "kd_loss_type": KD_LOSS_TYPE,
                "temperature": TEMPERATURE,
                "temperature_used": KD_LOSS_TYPE == "kl",
                "ce_warmup_epochs": CE_WARMUP_EPOCHS,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved paired text KD model: {best_dir.resolve()}")
    print(f"Saved outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
