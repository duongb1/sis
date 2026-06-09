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

from sislib.data.labels import EXCEL_MULTICLASS_LABELS


def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3-class MRI case-level classifier.")
    parser.add_argument("--folds-csv", default="mri_3class_folds.csv")
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--out", default="outputs/mri_3class/fold_0")
    parser.add_argument("--model", choices=["resnet18", "resnet34"], default="resnet18")
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
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_rows(path, fold_index, split):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if int(row["fold"]) == fold_index and row["split"] == split]


def case_images(image_dir):
    root = Path(image_dir)
    paths = []
    for subdir in ["DWI", "ADC"]:
        folder = root / subdir
        if folder.is_dir():
            paths.extend(sorted(folder.glob("*.JPG")))
    if not paths:
        paths = sorted(root.rglob("*.JPG"))
    return paths


class MRICaseDataset(Dataset):
    def __init__(self, rows, image_size=224, max_images_per_case=16, train=False, seed=42):
        self.rows = list(rows)
        self.max_images_per_case = max_images_per_case
        self.train = train
        self.rng = random.Random(seed)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=5),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

    def __len__(self):
        return len(self.rows)

    def _select_images(self, paths):
        if len(paths) <= self.max_images_per_case:
            return paths
        if self.train:
            return sorted(self.rng.sample(paths, self.max_images_per_case))
        step = len(paths) / self.max_images_per_case
        indices = [min(len(paths) - 1, int(i * step + step / 2)) for i in range(self.max_images_per_case)]
        return [paths[index] for index in indices]

    def __getitem__(self, index):
        row = self.rows[index]
        paths = self._select_images(case_images(row["image_dir"]))
        if not paths:
            raise FileNotFoundError(f"No JPG images found in {row['image_dir']}")
        images = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            images.append(self.transform(image))
        return {
            "images": torch.stack(images),
            "label": int(row["label_3class_id"]),
            "case_id": row["case_id"],
            "mabn": row["MABN"],
            "image_count": len(paths),
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
    def __init__(self, backbone_name="resnet18", num_classes=3, pretrained=False):
        super().__init__()
        if backbone_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            backbone = models.resnet34(weights=weights)
        else:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
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


def class_weights(rows):
    counts = Counter(int(row["label_3class_id"]) for row in rows)
    total = sum(counts.values())
    weights = []
    for class_id in range(len(EXCEL_MULTICLASS_LABELS)):
        weights.append(total / (len(EXCEL_MULTICLASS_LABELS) * counts[class_id]))
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)
        logits = model(images, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []
    case_ids, mabns, image_counts = [], [], []
    for batch in loader:
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)
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
            "true_label": EXCEL_MULTICLASS_LABELS[true],
            "pred_label": EXCEL_MULTICLASS_LABELS[pred],
            **{f"prob_{label}": prob[index] for index, label in enumerate(EXCEL_MULTICLASS_LABELS)},
        }
        for case_id, mabn, image_count, true, pred, prob in zip(case_ids, mabns, image_counts, y_true, y_pred, y_prob)
    ]
    return metrics, predictions


def compute_metrics(y_true, y_pred, y_prob):
    labels = list(range(len(EXCEL_MULTICLASS_LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": {
            EXCEL_MULTICLASS_LABELS[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in labels
        },
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=EXCEL_MULTICLASS_LABELS, zero_division=0, output_dict=True
        ),
    }
    binary_true = [1 if item == 0 else 0 for item in y_true]
    binary_pred = [1 if item == 0 else 0 for item in y_pred]
    binary_prob = [item[0] for item in y_prob]
    metrics["binary_i63"] = {
        "accuracy": accuracy_score(binary_true, binary_pred),
        "balanced_accuracy": balanced_accuracy_score(binary_true, binary_pred),
        "macro_f1": f1_score(binary_true, binary_pred, average="macro"),
        "i63_recall_sensitivity": precision_recall_fscore_support(binary_true, binary_pred, labels=[1], zero_division=0)[1][0],
        "non_i63_recall_specificity": precision_recall_fscore_support(binary_true, binary_pred, labels=[0], zero_division=0)[1][0],
    }
    if len(set(binary_true)) == 2:
        metrics["binary_i63"]["auc"] = roc_auc_score(binary_true, binary_prob)
    else:
        metrics["binary_i63"]["auc"] = math.nan
    return metrics


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

    train_rows = read_rows(args.folds_csv, args.fold_index, "train")
    val_rows = read_rows(args.folds_csv, args.fold_index, "val")
    test_rows = read_rows(args.folds_csv, args.fold_index, "test")
    datasets = {
        "train": MRICaseDataset(train_rows, args.image_size, args.max_images_per_case, train=True, seed=args.seed),
        "val": MRICaseDataset(val_rows, args.image_size, args.max_images_per_case, train=False, seed=args.seed),
        "test": MRICaseDataset(test_rows, args.image_size, args.max_images_per_case, train=False, seed=args.seed),
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

    model = CaseMeanPoolCNN(args.model, num_classes=len(EXCEL_MULTICLASS_LABELS), pretrained=args.pretrained).to(device)
    weights = None if args.no_class_weight else class_weights(train_rows).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    history = []
    best_val = -1.0
    best_path = out / "best.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics, _ = evaluate(model, loaders["val"], criterion, device)
        entry = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}}
        history.append(entry)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} val_bal_acc={val_metrics['balanced_accuracy']:.4f}"
        )
        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            torch.save({"model": model.state_dict(), "args": vars(args), "labels": EXCEL_MULTICLASS_LABELS}, best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    final = {"fold": args.fold_index, "device": str(device), "labels": EXCEL_MULTICLASS_LABELS, "history": history}
    for split in ["train", "val", "test"]:
        metrics, predictions = evaluate(model, loaders[split], criterion, device)
        final[split] = metrics
        write_predictions(out / f"predictions_{split}.csv", predictions)
    with (out / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final, handle, ensure_ascii=False, indent=2)
    print(f"wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
