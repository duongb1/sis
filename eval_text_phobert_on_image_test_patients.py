import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_TO_ID = {"khong": 0, "co": 1}
ID_TO_LABEL = {0: "khong", 1: "co"}


# Kaggle config. Image test layout:
# /kaggle/input/datasets/duongb/cthsis/images/test/{co,khong}/{patient_id}/{patient_id}.txt
IMAGE_ROOT = "/kaggle/input/datasets/duongb/cthsis/images"
MODEL_DIR = "/kaggle/working/text_phobert_classifier/best_auc_phobert"
OUTPUT_DIR = "/kaggle/working/text_phobert_classifier/image_test_patients"

LABELS = ["co", "khong"]
MAX_LENGTH = 512
BATCH_SIZE = 16
NUM_WORKERS = 0
THRESHOLD = 0.5
FORCE_CPU = False
USE_MULTI_GPU = True


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


def resolve_max_length(model, requested_max_length):
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is None:
        return requested_max_length
    return min(requested_max_length, max_positions - 2)


def collect_image_test_text_records(image_root):
    image_root = Path(image_root)
    if not (image_root / "test").exists() and (image_root / "lan1-full" / "test").exists():
        image_root = image_root / "lan1-full"

    records = []
    missing_text = []
    for label in LABELS:
        label_dir = image_root / "test" / label
        if not label_dir.exists():
            continue

        for patient_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
            text_path = patient_dir / f"{patient_dir.name}.txt"
            patient_id = f"test/{label}/{patient_dir.name}"
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
                    "label": LABEL_TO_ID[label],
                    "label_name": label,
                    "text_path": str(text_path),
                    "text": text,
                }
            )

    return records, missing_text, image_root


class TextPatientDataset(Dataset):
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
def evaluate(model, loader, device, threshold=THRESHOLD):
    model.eval()
    total_loss = 0.0
    total_count = 0
    patient_ids = []
    labels_all = []
    probs_all = []

    for batch in tqdm(loader, desc="Evaluating image-test text patients"):
        inputs, batch_patient_ids = batch_to_device(batch, device)
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
        "threshold": threshold,
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
        writer = csv.DictWriter(f, fieldnames=["patient_id", "label", "label_name", "text_path"])
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in ["patient_id", "label", "label_name", "text_path"]})


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(FORCE_CPU)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    max_length = resolve_max_length(model, MAX_LENGTH)
    if max_length != MAX_LENGTH:
        print(f"Requested MAX_LENGTH={MAX_LENGTH}, using {max_length}.")
    model = wrap_model_for_device(model, device)

    records, missing_text, image_root = collect_image_test_text_records(IMAGE_ROOT)
    if missing_text:
        (output_dir / "missing_text_patients.txt").write_text("\n".join(missing_text), encoding="utf-8")
    if not records:
        raise RuntimeError("No image-test patient text records found.")

    print(f"Image root: {image_root.resolve()}")
    print(f"Image-test text patients: {len(records)}")
    print(f"Missing/empty text patients: {len(missing_text)}")

    loader = DataLoader(
        TextPatientDataset(records, tokenizer, max_length),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    metrics, patient_ids, labels, probs, preds = evaluate(model, loader, device, THRESHOLD)
    metrics = rounded_metrics(metrics)
    save_predictions(output_dir / "image_test_text_predictions.csv", patient_ids, labels, probs, preds)
    save_dataset_records(output_dir / "image_test_text_records.csv", records)
    with open(output_dir / "image_test_text_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Metrics: {metrics}")
    print("Confusion matrix labels=[khong, co]:")
    print(confusion_matrix(labels, preds, labels=[0, 1]))
    print(f"Saved outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
