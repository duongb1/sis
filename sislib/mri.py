from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

from .common import LABELS, LABEL_TO_ID, SPLITS


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resolve_image_root(root, variant="lan1-full"):
    root = Path(root)
    if (root / "train").exists():
        return root
    if (root / variant / "train").exists():
        return root / variant
    raise FileNotFoundError(f"Cannot find split folders under {root}.")


def adc_dwi_zero(adc, dwi):
    if adc.size != dwi.size:
        dwi = dwi.resize(adc.size, Image.BILINEAR)
    adc_arr = np.asarray(adc, dtype=np.float32)
    dwi_arr = np.asarray(dwi, dtype=np.float32)
    zero = np.zeros_like(adc_arr)
    return Image.fromarray(np.clip(np.stack([adc_arr, dwi_arr, zero], axis=-1), 0, 255).astype(np.uint8))


def collect_pairs(root, max_per_class=None, splits=SPLITS):
    root = resolve_image_root(root)
    records = []
    for split in splits:
        by_label = defaultdict(list)
        for label in LABELS:
            label_dir = root / split / label
            if not label_dir.exists():
                continue
            for patient_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
                adc_dir, dwi_dir = patient_dir / "ADC", patient_dir / "DWI"
                adc = sorted(p for p in adc_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("._")) if adc_dir.exists() else []
                dwi = sorted(p for p in dwi_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("._")) if dwi_dir.exists() else []
                n = min(len(adc), len(dwi))
                if n == 0:
                    continue
                by_label[label].append(
                    {
                        "id": f"{split}/{label}/{patient_dir.name}",
                        "split": split,
                        "label": LABEL_TO_ID[label],
                        "label_name": label,
                        "pairs": list(zip(adc[:n], dwi[:n])),
                    }
                )
        for label in LABELS:
            rows = by_label[label]
            records.extend(rows[:max_per_class] if max_per_class is not None else rows)
    return records, root


class MRIDataset(Dataset):
    def __init__(self, records, transform):
        self.samples = []
        self.transform = transform
        for row in records:
            for adc, dwi in row["pairs"]:
                self.samples.append({"adc": adc, "dwi": dwi, "id": row["id"], "label": row["label"]})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        adc = Image.open(row["adc"]).convert("L")
        dwi = Image.open(row["dwi"]).convert("L")
        return self.transform(adc_dwi_zero(adc, dwi)), torch.tensor(row["label"], dtype=torch.float32), row["id"]


def mri_transforms(size):
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def resnet50_binary():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
