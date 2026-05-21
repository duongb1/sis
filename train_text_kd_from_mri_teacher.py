import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_TO_ID = {"khong": 0, "co": 1}
ID_TO_LABEL = {0: "khong", 1: "co"}


# Kaggle config. Edit these values in a cell if needed.
DATA_ROOT = "/kaggle/input/datasets/duongbui/siscth"
MRI_TEACHER_CKPT = "/kaggle/working/mri_classifier/best_auc_model.pt"
OUTPUT_DIR = "/kaggle/working/text_kd_from_correct_mri_teacher"

TEXT_MODEL_NAME = "vinai/phobert-base"
TEXT_STUDENT_INIT = "/kaggle/working/text_phobert_classifier/best_auc_phobert"
SPLITS = ["train", "val", "test"]
LABELS = ["co", "khong"]

MAX_LENGTH = 512
BATCH_SIZE_TEXT = 16
BATCH_SIZE_MRI = 64
EPOCHS = 3
LR = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
TEMPERATURE = 2.0
ALPHA_KD = 0.05  # total loss = (1-alpha)*CE + alpha*KD
THRESHOLD = 0.5
SEED = 42
NUM_WORKERS = 0
FORCE_CPU = False
USE_MULTI_GPU = True
GRADIENT_ACCUMULATION_STEPS = 1
MAX_PATIENTS_PER_CLASS = None

# KD variants.
# "binary": distill MRI sigmoid probability into text logit difference.
# "kl": classic 2-logit KL distillation.
KD_LOSS_TYPE = "binary"

# KD only uses MRI teacher signal when the MRI teacher predicts the true label.
TEACHER_CORRECT_ONLY = True
CE_WARMUP_EPOCHS = 0
BEST_MODEL_METRIC = "auc"

# MRI teacher patient aggregation: "mean", "max", or "topk_mean".
TEACHER_AGG = "mean"
TEACHER_TOPK_RATIO = 0.2

# Negative control: randomly reassign teacher signals among train patients only.
SHUFFLE_TEACHER = False


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_image_root(image_root, variant="lan1-full"):
    image_root = Path(image_root)
    if not image_root.exists() and image_root.name == "images":
        singular = image_root.with_name("image")
        if singular.exists():
            image_root = singular
    if (image_root / "train").exists():
        return image_root
    if (image_root / variant / "train").exists():
        return image_root / variant
    raise FileNotFoundError(f"Cannot find image split folders under {image_root}.")


def resolve_text_root(text_root):
    text_root = Path(text_root)
    if not text_root.exists() and text_root.name == "texts":
        singular = text_root.with_name("text")
        if singular.exists():
            text_root = singular
    for candidate in [text_root / "lan1" / "txt_nonFilter", text_root / "txt_nonFilter", text_root]:
        if (candidate / "train").exists() or (candidate / "test").exists():
            return candidate
    raise FileNotFoundError(f"Cannot find text split folders under {text_root}.")


def get_device(force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")

    supported_sms = getattr(torch.cuda, "get_arch_list", lambda: [])()
    for index in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(index)
        device_name = torch.cuda.get_device_name(index)
        current_sm = f"sm_{major}{minor}"
        if supported_sms and current_sm not in supported_sms:
            print(
                f"CUDA device {index} ({device_name}) has capability {current_sm}, but this PyTorch "
                f"build supports {supported_sms}. Falling back to CPU."
            )
            return torch.device("cpu")

    print(f"Using CUDA devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    return torch.device("cuda")


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def wrap_model_for_device(model, device):
    model = model.to(device)
    if device.type == "cuda" and USE_MULTI_GPU and torch.cuda.device_count() > 1:
        ids = list(range(torch.cuda.device_count()))
        print(f"Using DataParallel on GPU ids: {ids}")
        model = nn.DataParallel(model, device_ids=ids)
    return model


def build_adc_dwi_zero_image(adc, dwi):
    if adc.size != dwi.size:
        dwi = dwi.resize(adc.size, Image.BILINEAR)
    adc_arr = np.asarray(adc, dtype=np.float32)
    dwi_arr = np.asarray(dwi, dtype=np.float32)
    zero_arr = np.zeros_like(adc_arr)
    rgb = np.stack([adc_arr, dwi_arr, zero_arr], axis=-1)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def collect_patient_records(data_root):
    data_root = Path(data_root)
    records = []
    missing_text = []

    for split in SPLITS:
        by_label = defaultdict(list)
        for label in LABELS:
            label_dir = data_root / split / label
            if not label_dir.exists():
                continue
            for patient_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
                adc_dir = patient_dir / "ADC"
                dwi_dir = patient_dir / "DWI"
                adc_paths = sorted(
                    p for p in adc_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("._")
                ) if adc_dir.exists() else []
                dwi_paths = sorted(
                    p for p in dwi_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("._")
                ) if dwi_dir.exists() else []
                pair_count = min(len(adc_paths), len(dwi_paths))
                if pair_count == 0:
                    continue

                text_path = patient_dir / f"{patient_dir.name}.txt"
                patient_id = f"{split}/{label}/{patient_dir.name}"
                if not text_path.exists():
                    missing_text.append(patient_id)
                    continue

                by_label[label].append(
                    {
                        "patient_id": patient_id,
                        "split": split,
                        "label": LABEL_TO_ID[label],
                        "label_name": label,
                        "text_path": str(text_path),
                        "text": read_text_file(text_path).strip(),
                        "pairs": list(zip(adc_paths[:pair_count], dwi_paths[:pair_count])),
                    }
                )

        for label in LABELS:
            selected = by_label[label]
            if MAX_PATIENTS_PER_CLASS is not None:
                selected = selected[:MAX_PATIENTS_PER_CLASS]
            records.extend(selected)

    return records, missing_text, data_root


def read_text_file(path):
    for encoding in ("utf-8", "utf-8-sig", "cp1258", "latin-1"):
        try:
            return Path(path).read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return Path(path).read_text(encoding="utf-8", errors="ignore")


class MRIPairDataset(Dataset):
    def __init__(self, records, transform):
        self.samples = []
        self.transform = transform
        for record in records:
            for adc_path, dwi_path in record["pairs"]:
                self.samples.append(
                    {
                        "adc_path": adc_path,
                        "dwi_path": dwi_path,
                        "patient_id": record["patient_id"],
                        "label": record["label"],
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        adc = Image.open(sample["adc_path"]).convert("L")
        dwi = Image.open(sample["dwi_path"]).convert("L")
        image = build_adc_dwi_zero_image(adc, dwi)
        image = self.transform(image)
        return image, sample["patient_id"], sample["label"]


class TextKDDataset(Dataset):
    def __init__(self, records, tokenizer, max_length, teacher_logits, teacher_correct):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.teacher_logits = teacher_logits
        self.teacher_correct = teacher_correct

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
        item["teacher_logits"] = torch.tensor(self.teacher_logits[record["patient_id"]], dtype=torch.float32)
        item["teacher_correct"] = torch.tensor(self.teacher_correct[record["patient_id"]], dtype=torch.bool)
        item["patient_id"] = record["patient_id"]
        return item


def build_mri_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_mri_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def clean_state_dict(state_dict):
    if any(key.startswith("module.") for key in state_dict):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def load_mri_teacher(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    state_dict = clean_state_dict(state_dict)
    model = build_mri_model()
    model.load_state_dict(state_dict)
    model = wrap_model_for_device(model, device)
    model.eval()
    return model


@torch.no_grad()
def compute_teacher_logits(records, teacher, device):
    loader = DataLoader(
        MRIPairDataset(records, build_mri_transform()),
        batch_size=BATCH_SIZE_MRI,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    patient_logits = defaultdict(list)
    patient_labels = {}

    for images, patient_ids, labels in tqdm(loader, desc="Computing MRI teacher logits"):
        images = images.to(device, non_blocking=True)
        logits = teacher(images).squeeze(1).detach().cpu().numpy()
        for logit, patient_id, label in zip(logits, patient_ids, labels):
            patient_logits[patient_id].append(float(logit))
            patient_labels[patient_id] = int(label)

    result = {}
    teacher_correct = {}
    labels = []
    probs = []
    patient_ids = []
    for patient_id, logits in patient_logits.items():
        sorted_logits = sorted(logits)
        if TEACHER_AGG == "mean":
            logit_co = float(np.mean(sorted_logits))
        elif TEACHER_AGG == "max":
            logit_co = float(np.max(sorted_logits))
        elif TEACHER_AGG == "topk_mean":
            k = max(1, int(np.ceil(len(sorted_logits) * TEACHER_TOPK_RATIO)))
            logit_co = float(np.mean(sorted_logits[-k:]))
        else:
            raise ValueError(f"Unsupported TEACHER_AGG: {TEACHER_AGG}")
        prob_co = 1.0 / (1.0 + np.exp(-logit_co))
        pred = int(prob_co >= 0.5)
        label = patient_labels[patient_id]
        result[patient_id] = [-logit_co / 2.0, logit_co / 2.0]
        teacher_correct[patient_id] = pred == label
        patient_ids.append(patient_id)
        labels.append(label)
        probs.append(prob_co)

    labels = np.array(labels, dtype=np.int64)
    probs = np.array(probs, dtype=np.float32)
    preds = (probs >= 0.5).astype(np.int64)
    confs = np.maximum(probs, 1.0 - probs)
    correct_count = int(np.sum(preds == labels))
    stats = {
        "teacher_acc_patient_level": float(accuracy_score(labels, preds)),
        "teacher_f1_patient_level": float(f1_score(labels, preds, zero_division=0)),
        "teacher_auc_patient_level": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else float("nan"),
        "mean_teacher_conf": float(np.mean(confs)),
        "teacher_num_patients": int(len(labels)),
        "teacher_correct_patients": correct_count,
        "teacher_wrong_patients": int(len(labels) - correct_count),
        "teacher_correct_ratio": float(correct_count / max(len(labels), 1)),
        "teacher_agg": TEACHER_AGG,
        "teacher_correct_only": TEACHER_CORRECT_ONLY,
    }
    return result, teacher_correct, stats


def merge_dicts(*dicts):
    merged = {}
    for item in dicts:
        merged.update(item)
    return merged


def shuffle_teacher_for_train(teacher_logits, teacher_correct, train_records, seed):
    rng = random.Random(seed)
    ids = [record["patient_id"] for record in train_records]
    shuffled_ids = ids[:]
    rng.shuffle(shuffled_ids)

    shuffled_logits = dict(teacher_logits)
    shuffled_correct = dict(teacher_correct)
    for target_id, source_id in zip(ids, shuffled_ids):
        shuffled_logits[target_id] = teacher_logits[source_id]
        shuffled_correct[target_id] = teacher_correct[source_id]
    return shuffled_logits, shuffled_correct


def batch_to_device(batch, device):
    patient_ids = batch.pop("patient_id")
    inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    return inputs, patient_ids


def kd_loss(student_logits, teacher_logits, teacher_correct, labels, epoch):
    ce_per_sample = F.cross_entropy(student_logits, labels, reduction="none")
    teacher_probs = F.softmax(teacher_logits, dim=-1)

    if epoch <= CE_WARMUP_EPOCHS or ALPHA_KD <= 0:
        zero = torch.zeros((), device=student_logits.device)
        return ce_per_sample.mean(), ce_per_sample.mean().detach(), zero

    if KD_LOSS_TYPE == "binary":
        teacher_p_co = teacher_probs[:, 1].detach()
        student_logit_co = student_logits[:, 1] - student_logits[:, 0]
        kd_per_sample = F.binary_cross_entropy_with_logits(
            student_logit_co,
            teacher_p_co,
            reduction="none",
        )
    elif KD_LOSS_TYPE == "kl":
        kd_per_sample = F.kl_div(
            F.log_softmax(student_logits / TEMPERATURE, dim=-1),
            F.softmax(teacher_logits / TEMPERATURE, dim=-1),
            reduction="none",
        ).sum(dim=-1) * (TEMPERATURE ** 2)
    else:
        raise ValueError(f"Unsupported KD_LOSS_TYPE: {KD_LOSS_TYPE}")

    if TEACHER_CORRECT_ONLY:
        sample_alpha = torch.where(
            teacher_correct,
            torch.full_like(ce_per_sample, ALPHA_KD),
            torch.zeros_like(ce_per_sample),
        )
    else:
        sample_alpha = torch.full_like(ce_per_sample, ALPHA_KD)

    loss_per_sample = (1.0 - sample_alpha) * ce_per_sample + sample_alpha * kd_per_sample
    if teacher_correct.any():
        kd_log = kd_per_sample[teacher_correct].mean().detach()
    else:
        kd_log = torch.zeros((), device=student_logits.device)
    return loss_per_sample.mean(), ce_per_sample.mean().detach(), kd_log


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_count = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="Training KD", leave=False), start=1):
        inputs, _ = batch_to_device(batch, device)
        teacher_logits = inputs.pop("teacher_logits")
        teacher_correct = inputs.pop("teacher_correct")
        labels = inputs["labels"]

        use_amp = device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(**inputs)
            loss, ce, kd = kd_loss(outputs.logits, teacher_logits, teacher_correct, labels, epoch)
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
def evaluate(model, loader, device, threshold=None):
    model.eval()
    total_loss = 0.0
    total_count = 0
    ids, labels_all, probs_all = [], [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        inputs, patient_ids = batch_to_device(batch, device)
        teacher_logits = inputs.pop("teacher_logits")
        teacher_correct = inputs.pop("teacher_correct")
        del teacher_logits
        del teacher_correct
        labels = inputs["labels"]
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs.logits, labels)
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1]

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        ids.extend(patient_ids)
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        probs_all.extend(probs.detach().cpu().numpy().tolist())

    labels = np.array(labels_all, dtype=np.int64)
    probs = np.array(probs_all, dtype=np.float32)
    if threshold is None:
        threshold = THRESHOLD
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
    return metrics, ids, labels, probs, preds


def split_records(records, split):
    return [r for r in records if r["split"] == split]


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
        for row in records:
            writer.writerow({k: row[k] for k in ["patient_id", "split", "label", "label_name", "text_path"]})


def main():
    seed_everything(SEED)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(FORCE_CPU)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    records, missing_text, data_root = collect_patient_records(DATA_ROOT)
    if missing_text:
        (output_dir / "missing_text_patients.txt").write_text("\n".join(missing_text), encoding="utf-8")

    train_records = split_records(records, "train")
    val_records = split_records(records, "val")
    test_records = split_records(records, "test")
    if not train_records or not val_records:
        raise RuntimeError("Need non-empty train and val records.")

    print(f"Data root: {data_root.resolve()}")
    print(f"Records with MRI+text: {len(records)}")
    print(f"Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}")

    teacher = load_mri_teacher(MRI_TEACHER_CKPT, device)
    teacher_logits_train, teacher_correct_train, teacher_stats_train = compute_teacher_logits(train_records, teacher, device)
    teacher_logits_val, teacher_correct_val, teacher_stats_val = compute_teacher_logits(val_records, teacher, device)
    if test_records:
        teacher_logits_test, teacher_correct_test, teacher_stats_test = compute_teacher_logits(test_records, teacher, device)
    else:
        teacher_logits_test, teacher_correct_test, teacher_stats_test = {}, {}, {}

    teacher_logits = merge_dicts(teacher_logits_train, teacher_logits_val, teacher_logits_test)
    teacher_correct = merge_dicts(teacher_correct_train, teacher_correct_val, teacher_correct_test)
    teacher_stats = {
        "train": rounded_metrics(teacher_stats_train),
        "val": rounded_metrics(teacher_stats_val),
        "test": rounded_metrics(teacher_stats_test) if teacher_stats_test else None,
    }
    print(f"Teacher stats by split: {teacher_stats}")
    with open(output_dir / "teacher_stats.json", "w", encoding="utf-8") as f:
        json.dump(teacher_stats, f, ensure_ascii=False, indent=2)
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if SHUFFLE_TEACHER:
        teacher_logits, teacher_correct = shuffle_teacher_for_train(
            teacher_logits,
            teacher_correct,
            train_records,
            SEED,
        )

    tokenizer = AutoTokenizer.from_pretrained(TEXT_STUDENT_INIT, use_fast=False)
    student = AutoModelForSequenceClassification.from_pretrained(
        TEXT_STUDENT_INIT,
        num_labels=2,
        id2label={0: "khong", 1: "co"},
        label2id={"khong": 0, "co": 1},
    )
    student = wrap_model_for_device(student, device)

    train_loader = DataLoader(
        TextKDDataset(train_records, tokenizer, MAX_LENGTH, teacher_logits, teacher_correct),
        batch_size=BATCH_SIZE_TEXT,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TextKDDataset(val_records, tokenizer, MAX_LENGTH, teacher_logits, teacher_correct),
        batch_size=BATCH_SIZE_TEXT,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        TextKDDataset(test_records, tokenizer, MAX_LENGTH, teacher_logits, teacher_correct),
        batch_size=BATCH_SIZE_TEXT,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    ) if test_records else None

    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS // max(GRADIENT_ACCUMULATION_STEPS, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_score = -1.0
    best_dir = output_dir / "best_text_student_kd"
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
            best_dir.mkdir(parents=True, exist_ok=True)
            unwrap_model(student).save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            save_predictions(output_dir / "val_predictions_best.csv", val_ids, val_labels, val_probs, val_preds)

    with open(output_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    save_dataset_records(output_dir / "dataset_records.csv", records)

    student = AutoModelForSequenceClassification.from_pretrained(best_dir)
    student = wrap_model_for_device(student, device)

    all_metrics = {}
    best_threshold = THRESHOLD

    for split, loader in [("val", val_loader), ("test", test_loader)]:
        if loader is None:
            continue
        metrics, ids, labels, probs, preds = evaluate(student, loader, device, best_threshold)
        all_metrics[split] = rounded_metrics(metrics)
        save_predictions(output_dir / f"{split}_predictions.csv", ids, labels, probs, preds)
        print(f"{split}: {rounded_metrics(metrics)}")
        print(confusion_matrix(labels, preds, labels=[0, 1]))

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    with open(output_dir / "kd_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "teacher": MRI_TEACHER_CKPT,
                "base_text_model": TEXT_MODEL_NAME,
                "student_init": TEXT_STUDENT_INIT,
                "temperature": TEMPERATURE,
                "temperature_used": KD_LOSS_TYPE == "kl",
                "alpha_kd": ALPHA_KD,
                "best_model_metric": BEST_MODEL_METRIC,
                "kd_loss_type": KD_LOSS_TYPE,
                "teacher_correct_only": TEACHER_CORRECT_ONLY,
                "shuffle_teacher": SHUFFLE_TEACHER,
                "ce_warmup_epochs": CE_WARMUP_EPOCHS,
                "teacher_agg": TEACHER_AGG,
                "teacher_topk_ratio": TEACHER_TOPK_RATIO,
                "best_threshold": round_float(best_threshold),
                "max_length": MAX_LENGTH,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved KD student: {best_dir.resolve()}")
    print(f"Saved outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
