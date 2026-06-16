import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

LABELS = ["khong", "co"]


def discover_mri_cases(image_root):
    root = Path(image_root)
    cases = []
    
    # Check if we have class subdirectories (old binary split structure)
    has_co = (root / "co").is_dir()
    has_khong = (root / "khong").is_dir()
    
    # Support nested split subdirectories (train/co, val/co, test/co)
    has_splits = any((root / s).is_dir() for s in ["train", "val", "test"])
    
    if has_splits:
        for split in ["train", "val", "test"]:
            for class_name, label_val in [("co", 1), ("khong", 0)]:
                class_dir = root / split / class_name
                if class_dir.is_dir():
                    for case_dir in class_dir.iterdir():
                        if case_dir.is_dir():
                            if (case_dir / "ADC").is_dir() and (case_dir / "DWI").is_dir():
                                cases.append({
                                    "case_id": case_dir.name,
                                    "label": label_val,
                                    "label_name": class_name,
                                    "image_dir": str(case_dir),
                                    "MABN": case_dir.name,
                                })
    elif has_co or has_khong:
        for class_name, label_val in [("co", 1), ("khong", 0)]:
            class_dir = root / class_name
            if class_dir.is_dir():
                for case_dir in class_dir.iterdir():
                    if case_dir.is_dir():
                        if (case_dir / "ADC").is_dir() and (case_dir / "DWI").is_dir():
                            cases.append({
                                "case_id": case_dir.name,
                                "label": label_val,
                                "label_name": class_name,
                                "image_dir": str(case_dir),
                                "MABN": case_dir.name,
                            })
    else:
        # Flat structure: image_root/stt (STT 1-1400)
        if root.is_dir():
            for case_dir in root.iterdir():
                if case_dir.is_dir() and case_dir.name.isdigit():
                    stt_val = int(case_dir.name)
                    # 700_co has STT 1-700
                    # 700_khong has STT 701-1400
                    if 1 <= stt_val <= 700:
                        class_name = "co"
                        label_val = 1
                    elif 701 <= stt_val <= 1400:
                        class_name = "khong"
                        label_val = 0
                    else:
                        continue
                    
                    if (case_dir / "ADC").is_dir() and (case_dir / "DWI").is_dir():
                        cases.append({
                            "case_id": case_dir.name,
                            "label": label_val,
                            "label_name": class_name,
                            "image_dir": str(case_dir),
                            "MABN": case_dir.name,
                        })
    return cases


def make_stratified_folds(cases, n_folds=5, seed=42):
    rng = random.Random(seed)
    pos_cases = [c for c in cases if c["label"] == 1]
    neg_cases = [c for c in cases if c["label"] == 0]
    rng.shuffle(pos_cases)
    rng.shuffle(neg_cases)
    
    folds = [[] for _ in range(n_folds)]
    for i, c in enumerate(pos_cases):
        folds[i % n_folds].append(c)
    for i, c in enumerate(neg_cases):
        folds[i % n_folds].append(c)
        
    assigned_cases = []
    for f_idx, fold_list in enumerate(folds):
        for c in fold_list:
            c_copy = dict(c)
            c_copy["fold"] = f_idx
            assigned_cases.append(c_copy)
    return assigned_cases


def get_splits_for_fold(cases, fold_index, n_folds=5):
    test_rows = [c for c in cases if c["fold"] == fold_index]
    val_rows = [c for c in cases if c["fold"] == (fold_index + 1) % n_folds]
    train_rows = [c for c in cases if c["fold"] not in (fold_index, (fold_index + 1) % n_folds)]
    return train_rows, val_rows, test_rows


DEFAULT_KAGGLE_MRI_ROOT = "/kaggle/input/datasets/duongbui/siscth/images"
mri_root_default = DEFAULT_KAGGLE_MRI_ROOT if Path(DEFAULT_KAGGLE_MRI_ROOT).exists() else "images"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a binary MRI case-level classifier.")
    parser.add_argument("--folds-csv", default="")
    parser.add_argument("--image-root", default=mri_root_default)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--out", default="mri_binary/fold_0")
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision ImageNet weights if available.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-images-per-case", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--no-dp", action="store_true", help="Disable DataParallel when multiple CUDA devices are available.")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def case_image_pairs(image_dir):
    root = Path(image_dir)
    adc_paths = sorted((root / "ADC").glob("*.JPG")) if (root / "ADC").is_dir() else []
    dwi_paths = sorted((root / "DWI").glob("*.JPG")) if (root / "DWI").is_dir() else []
    pair_count = min(len(adc_paths), len(dwi_paths))
    return list(zip(adc_paths[:pair_count], dwi_paths[:pair_count]))


def resolve_image_dir(row, image_root):
    if row.get("image_dir"):
        p = Path(row["image_dir"])
        if p.is_dir():
            return p
    label_name = row.get("label_name", "co" if row.get("label") == 1 else "khong")
    
    # Try flat directory first
    flat_path = Path(image_root) / str(row["case_id"])
    if flat_path.is_dir():
        return flat_path
        
    # Try binary directory
    binary_path = Path(image_root) / label_name / str(row["case_id"])
    if binary_path.is_dir():
        return binary_path
        
    # Try split directory
    for split in ["train", "val", "test"]:
        split_path = Path(image_root) / split / label_name / str(row["case_id"])
        if split_path.is_dir():
            return split_path
            
    return binary_path


class MRICaseDataset(Dataset):
    def __init__(self, rows, image_root, image_size=224, max_images_per_case=16, train=False, seed=42):
        self.rows = list(rows)
        self.image_root = image_root
        self.max_images_per_case = max_images_per_case
        self.train = train
        self.rng = random.Random(seed)
        self.resize = transforms.Resize((image_size, image_size))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.0], std=[0.229, 0.224, 1.0])
        if train:
            self.spatial_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=5)]
            )
        else:
            self.spatial_transform = None

    def __len__(self):
        return len(self.rows)

    def _select_pairs(self, pairs):
        if len(pairs) <= self.max_images_per_case:
            return pairs
        if self.train:
            return sorted(self.rng.sample(pairs, self.max_images_per_case))
        step = len(pairs) / self.max_images_per_case
        indices = [min(len(pairs) - 1, int(i * step + step / 2)) for i in range(self.max_images_per_case)]
        return [pairs[index] for index in indices]

    def _load_pair(self, adc_path, dwi_path):
        adc = self.resize(Image.open(adc_path).convert("L"))
        dwi = self.resize(Image.open(dwi_path).convert("L"))
        if self.spatial_transform is not None:
            seed = self.rng.randint(0, 2**32 - 1)
            random.seed(seed)
            torch.manual_seed(seed)
            adc = self.spatial_transform(adc)
            random.seed(seed)
            torch.manual_seed(seed)
            dwi = self.spatial_transform(dwi)
        adc_tensor = self.to_tensor(adc).squeeze(0)
        dwi_tensor = self.to_tensor(dwi).squeeze(0)
        zeros = torch.zeros_like(adc_tensor)
        return self.normalize(torch.stack([adc_tensor, dwi_tensor, zeros], dim=0))

    def __getitem__(self, index):
        row = self.rows[index]
        image_dir = resolve_image_dir(row, self.image_root)
        pairs = self._select_pairs(case_image_pairs(image_dir))
        if not pairs:
            raise FileNotFoundError(f"No paired ADC/DWI JPG images found in {image_dir}")
        images = []
        for adc_path, dwi_path in pairs:
            images.append(self._load_pair(adc_path, dwi_path))
        return {
            "images": torch.stack(images),
            "label": int(row["label"]),
            "case_id": row["case_id"],
            "mabn": row.get("MABN", row["case_id"]),
            "image_count": len(pairs),
        }


def collate_cases(batch):
    images = [item["images"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    lengths = torch.tensor([item.shape[0] for item in images], dtype=torch.long)
    return {
        "images": torch.cat(images, dim=0),
        "labels": labels,
        "lengths": lengths,
        "case_id": [item["case_id"] for item in batch],
        "mabn": [item["mabn"] for item in batch],
        "image_count": [item["image_count"] for item in batch],
    }


class CaseMeanPoolCNN(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, images, lengths):
        features = self.backbone(images)
        pooled = []
        offset = 0
        for length in lengths.tolist():
            pooled.append(features[offset : offset + length].mean(dim=0))
            offset += length
        case_features = torch.stack(pooled, dim=0)
        return self.classifier(case_features)


class CaseMeanPoolDP(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        from utils.common import AutocastDPWrapper
        self.backbone = nn.DataParallel(AutocastDPWrapper(base_model.backbone))
        self.classifier = base_model.classifier

    def forward(self, images, lengths):
        features = self.backbone(images)
        pooled = []
        offset = 0
        for length in lengths.tolist():
            pooled.append(features[offset : offset + length].mean(dim=0))
            offset += length
        case_features = torch.stack(pooled, dim=0)
        return self.classifier(case_features)


def class_weights(rows):
    counts = Counter(int(row["label"]) for row in rows)
    total = sum(counts.values())
    weights = []
    for class_id in range(2):
        count = counts.get(class_id, 0)
        weights.append(total / (2 * count) if count > 0 else 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def autocast_context(device, precision):
    return torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda" and precision == "fp16"))


def run_epoch(model, loader, criterion, optimizer, scaler, device, precision):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)
        with autocast_context(device, precision):
            logits = model(images, lengths)
            loss = criterion(logits, labels)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def evaluate(model, loader, criterion, device, precision):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []
    case_ids, mabns, image_counts = [], [], []
    for batch in loader:
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)
        with autocast_context(device, precision):
            logits = model(images, lengths)
            loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        total_loss += loss.item() * labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(probs.argmax(dim=1).cpu().tolist())
        y_prob.extend(probs.cpu().tolist())
        case_ids.extend(batch["case_id"])
        mabns.extend(batch["mabn"])
        image_counts.extend(batch["image_count"])
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    predictions = [
        {
            "case_id": case_id,
            "MABN": mabn,
            "image_count": image_count,
            "true_label": LABELS[true],
            "pred_label": LABELS[pred],
            **{f"prob_{label}": prob[index] for index, label in enumerate(LABELS)},
        }
        for case_id, mabn, image_count, true, pred, prob in zip(case_ids, mabns, image_counts, y_true, y_pred, y_prob)
    ]
    return metrics, predictions


def compute_metrics(y_true, y_pred, y_prob):
    from utils.metrics import cls_metrics
    return cls_metrics(
        y_true,
        np.array(y_prob),
        y_pred,
        threshold=0.5,
        label_names=LABELS,
        binary_positive_label="co",
    )


def write_predictions(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    seed_everything(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    cases = discover_mri_cases(args.image_root)
    if not cases:
        raise ValueError(f"No MRI cases discovered under {args.image_root}")
    cases_with_folds = make_stratified_folds(cases, n_folds=5, seed=args.seed)
    train_rows, val_rows, test_rows = get_splits_for_fold(cases_with_folds, args.fold_index, n_folds=5)

    print(f"Discovered {len(cases)} MRI cases total in {args.image_root}")
    print(f"Fold {args.fold_index} split sizes: Train={len(train_rows)}, Val={len(val_rows)}, Test={len(test_rows)}")

    datasets = {
        "train": MRICaseDataset(train_rows, args.image_root, args.image_size, args.max_images_per_case, train=True, seed=args.seed),
        "val": MRICaseDataset(val_rows, args.image_root, args.image_size, args.max_images_per_case, train=False, seed=args.seed),
        "test": MRICaseDataset(test_rows, args.image_root, args.image_size, args.max_images_per_case, train=False, seed=args.seed),
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=(split == "train"),
            num_workers=args.workers,
            collate_fn=collate_cases,
        )
        for split, dataset in datasets.items()
    }

    model = CaseMeanPoolCNN(num_classes=2, pretrained=args.pretrained).to(device)
    cuda_devices = torch.cuda.device_count() if device.type == "cuda" else 0
    if cuda_devices > 1 and not args.no_dp:
        model = CaseMeanPoolDP(model)
    print(f"device={device} cuda_devices={cuda_devices} precision={args.precision} data_parallel={isinstance(model, CaseMeanPoolDP)}")
    weights = None if args.no_class_weight else class_weights(train_rows).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.precision == "fp16"))

    history = []
    best_val = -1.0
    best_path = out / "best.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, loaders["train"], criterion, optimizer, scaler, device, args.precision)
        val_metrics, _ = evaluate(model, loaders["val"], criterion, device, args.precision)
        entry = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}}
        history.append(entry)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_accuracy={val_metrics['accuracy']:.4f}"
        )
        if val_metrics["f1"] > best_val:
            best_val = val_metrics["f1"]
            torch.save({"model": model.state_dict(), "args": vars(args), "labels": LABELS}, best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    final = {"fold": args.fold_index, "device": str(device), "labels": LABELS, "history": history}
    for split in ["train", "val", "test"]:
        metrics, predictions = evaluate(model, loaders[split], criterion, device, args.precision)
        final[split] = metrics
        write_predictions(out / f"predictions_{split}.csv", predictions)
    with (out / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final, handle, ensure_ascii=False, indent=2)
    print(f"wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
