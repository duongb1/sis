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


# Kaggle config. Dataset layout:
# /kaggle/input/datasets/duongb/cthsis/texts/{train,val,test}/{co,khong}/*.txt
DATA_ROOT = "/kaggle/input/datasets/duongb/cthsis/texts"
OUTPUT_DIR = "/kaggle/working/text_phobert_classifier"

TEXT_MODEL_NAME = "vinai/phobert-base"
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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def collect_text_records(data_root):
    data_root = Path(data_root)
    records = []
    skipped = []

    for split in SPLITS:
        for label in LABELS:
            label_dir = data_root / split / label
            if not label_dir.exists():
                continue

            for text_path in sorted(label_dir.glob("*.txt")):
                if text_path.name.startswith("._"):
                    skipped.append(str(text_path))
                    continue

                text = read_text_file(text_path).strip()
                if not text:
                    skipped.append(str(text_path))
                    continue

                records.append(
                    {
                        "sample_id": f"{split}/{label}/{text_path.stem}",
                        "split": split,
                        "label": LABEL_TO_ID[label],
                        "label_name": label,
                        "text_path": str(text_path),
                        "text": text,
                    }
                )

    return records, skipped, data_root


def split_records(records, split):
    return [record for record in records if record["split"] == split]


class TextDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

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
        item["sample_id"] = record["sample_id"]
        return item


def batch_to_device(batch, device):
    sample_ids = batch.pop("sample_id")
    inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    return inputs, sample_ids


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


def build_model():
    return AutoModelForSequenceClassification.from_pretrained(
        TEXT_MODEL_NAME,
        num_labels=2,
        id2label={0: "khong", 1: "co"},
        label2id={"khong": 0, "co": 1},
    )


def resolve_max_length(model, requested_max_length):
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is None:
        return requested_max_length
    return min(requested_max_length, max_positions - 2)


def get_loss(outputs, labels):
    loss = outputs.loss if outputs.loss is not None else F.cross_entropy(outputs.logits, labels)
    return loss.mean()


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0.0
    total_count = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="Training", leave=False), start=1):
        inputs, _ = batch_to_device(batch, device)
        labels = inputs["labels"]

        use_amp = device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(**inputs)
            loss = get_loss(outputs, labels)
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
        total_count += batch_size

    return float(total_loss / max(total_count, 1))


@torch.no_grad()
def evaluate(model, loader, device, threshold=THRESHOLD):
    model.eval()
    total_loss = 0.0
    total_count = 0
    sample_ids = []
    labels_all = []
    probs_all = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        inputs, batch_sample_ids = batch_to_device(batch, device)
        labels = inputs["labels"]
        outputs = model(**inputs)
        loss = get_loss(outputs, labels)
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1]

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        sample_ids.extend(batch_sample_ids)
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
        "num_samples": int(len(labels)),
        "num_co": int((labels == 1).sum()),
        "num_khong": int((labels == 0).sum()),
    }
    return metrics, sample_ids, labels, probs, preds


def save_predictions(path, sample_ids, labels, probs, preds):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "true_label", "true_name", "prob_co", "pred_label", "pred_name"],
        )
        writer.writeheader()
        for sample_id, label, prob, pred in zip(sample_ids, labels, probs, preds):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "true_label": int(label),
                    "true_name": ID_TO_LABEL[int(label)],
                    "prob_co": round_float(prob),
                    "pred_label": int(pred),
                    "pred_name": ID_TO_LABEL[int(pred)],
                }
            )


def save_dataset_records(path, records):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "split", "label", "label_name", "text_path"])
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in ["sample_id", "split", "label", "label_name", "text_path"]})


def save_model(model, tokenizer, path, epoch, metric_name, val_metrics, max_length):
    path.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(path)
    tokenizer.save_pretrained(path)
    with open(path / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "epoch": epoch,
                "best_model_metric": metric_name,
                "val_metrics": rounded_metrics(val_metrics),
                "text_model_name": TEXT_MODEL_NAME,
                "max_length": max_length,
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

    records, skipped, data_root = collect_text_records(DATA_ROOT)
    if skipped:
        (output_dir / "skipped_text_files.txt").write_text("\n".join(skipped), encoding="utf-8")

    train_records = split_records(records, "train")
    val_records = split_records(records, "val")
    test_records = split_records(records, "test")
    if not train_records or not val_records:
        raise RuntimeError("Need non-empty train and val records.")

    print(f"Data root: {data_root.resolve()}")
    print(f"Text samples: {len(records)}")
    print(f"Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}")

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, use_fast=False)
    model = build_model()
    effective_max_length = resolve_max_length(model, MAX_LENGTH)
    if effective_max_length != MAX_LENGTH:
        print(
            f"Requested MAX_LENGTH={MAX_LENGTH}, using {effective_max_length} because "
            f"{TEXT_MODEL_NAME} max_position_embeddings={model.config.max_position_embeddings}."
        )
    model = wrap_model_for_device(model, device)

    train_loader = DataLoader(
        TextDataset(train_records, tokenizer, effective_max_length),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TextDataset(val_records, tokenizer, effective_max_length),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        TextDataset(test_records, tokenizer, effective_max_length),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    ) if test_records else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    updates_per_epoch = int(np.ceil(len(train_loader) / max(GRADIENT_ACCUMULATION_STEPS, 1)))
    total_steps = max(1, updates_per_epoch * EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_score = -1.0
    best_dir = output_dir / "best_auc_phobert"
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device)
        val_metrics, val_ids, val_labels, val_probs, val_preds = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": round_float(train_loss),
            "val_loss": round_float(val_metrics["loss"]),
            "val_acc": round_float(val_metrics["accuracy"]),
            "val_f1": round_float(val_metrics["f1"]),
            "val_auc": round_float(val_metrics["auc"]),
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{EPOCHS} | train_loss={train_loss:.3f} | "
            f"val_loss={val_metrics['loss']:.3f} | val_acc={val_metrics['accuracy']:.3f} | "
            f"val_f1={val_metrics['f1']:.3f} | val_auc={val_metrics['auc']:.3f}"
        )

        score = val_metrics["auc"]
        if score > best_score:
            best_score = score
            save_model(model, tokenizer, best_dir, epoch, "auc", val_metrics, effective_max_length)
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
                "data_root": str(data_root),
                "text_model_name": TEXT_MODEL_NAME,
                "requested_max_length": MAX_LENGTH,
                "max_length": effective_max_length,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "warmup_ratio": WARMUP_RATIO,
                "threshold": THRESHOLD,
                "best_model_metric": "auc",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved best AUC PhoBERT model: {best_dir.resolve()}")
    print(f"Saved outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
