import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .common import LABELS, LABEL_TO_ID, SPLITS, read_text


def collect_large_text(root):
    root = Path(root)
    records, skipped = [], []
    for split in SPLITS:
        for label in LABELS:
            label_dir = root / split / label
            if not label_dir.exists():
                continue
            for path in sorted(label_dir.glob("*.txt")):
                if path.name.startswith("._"):
                    skipped.append(str(path))
                    continue
                text = read_text(path).strip()
                if not text:
                    skipped.append(str(path))
                    continue
                records.append(
                    {
                        "id": f"{split}/{label}/{path.stem}",
                        "split": split,
                        "label": LABEL_TO_ID[label],
                        "label_name": label,
                        "text_path": str(path),
                        "text": text,
                    }
                )
    return records, skipped, root


def resolve_image_root(root):
    root = Path(root)
    if (root / "train").exists():
        return root
    if (root / "lan1-full" / "train").exists():
        return root / "lan1-full"
    raise FileNotFoundError(f"Cannot find split folders under {root}.")


def collect_paired_text(image_root, splits=SPLITS):
    image_root = resolve_image_root(image_root)
    records, missing = [], []
    for split in splits:
        for label in LABELS:
            label_dir = image_root / split / label
            if not label_dir.exists():
                continue
            for patient_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
                path = patient_dir / f"{patient_dir.name}.txt"
                pid = f"{split}/{label}/{patient_dir.name}"
                if not path.exists():
                    missing.append(pid)
                    continue
                text = read_text(path).strip()
                if not text:
                    missing.append(pid)
                    continue
                records.append(
                    {
                        "id": pid,
                        "split": split,
                        "label": LABEL_TO_ID[label],
                        "label_name": label,
                        "text_path": str(path),
                        "text": text,
                    }
                )
    return records, missing, image_root


class TextDataset(Dataset):
    def __init__(self, records, tokenizer, max_len, teacher_logits=None):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.teacher_logits = teacher_logits

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(row["label"], dtype=torch.long)
        if self.teacher_logits is not None:
            item["teacher_logits"] = torch.tensor(self.teacher_logits[row["id"]], dtype=torch.float32)
        item["id"] = row["id"]
        return item


def save_records(path, records):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "split", "label", "label_name", "text_path"])
        writer.writeheader()
        for row in records:
            writer.writerow({k: row[k] for k in ["id", "split", "label", "label_name", "text_path"]})
