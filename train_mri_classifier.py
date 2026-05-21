import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_TO_ID = {"khong": 0, "co": 1}
ID_TO_LABEL = {0: "khong", 1: "co"}


# Kaggle notebook config. Edit these values in the first cell if needed.
IMAGE_ROOT = "/kaggle/input/datasets/duongbui/siscth"
OUTPUT_DIR = "/kaggle/working/mri_classifier"
SPLITS = ["train", "val", "test"]
LABELS = ["co", "khong"]
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 0
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
FORCE_CPU = False
USE_MULTI_GPU = True
MAX_PATIENTS_PER_CLASS = None
THRESHOLD = 0.5


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
    raise FileNotFoundError(
        f"Cannot find split folders under {image_root}. Expected {image_root / 'train'} "
        f"or {image_root / variant / 'train'}."
    )


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

    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(f"Using CUDA devices: {gpu_names}")
    return torch.device("cuda")


def wrap_model_for_device(model, device, use_multi_gpu=True):
    model = model.to(device)
    if device.type == "cuda" and use_multi_gpu and torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        print(f"Using DataParallel on GPU ids: {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)
    return model


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def build_adc_dwi_zero_image(adc, dwi):
    if adc.size != dwi.size:
        dwi = dwi.resize(adc.size, Image.BILINEAR)

    adc_arr = np.asarray(adc, dtype=np.float32)
    dwi_arr = np.asarray(dwi, dtype=np.float32)
    zero_arr = np.zeros_like(adc_arr)
    rgb = np.stack([adc_arr, dwi_arr, zero_arr], axis=-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)


def collect_patient_pairs(image_root, splits, labels, max_patients_per_class=None):
    image_root = resolve_image_root(image_root)
    records = []

    for split in splits:
        split_records = []
        by_label = defaultdict(list)

        for label in labels:
            label_dir = image_root / split / label
            if not label_dir.exists():
                continue

            for patient_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
                adc_paths = sorted(
                    p for p in (patient_dir / "ADC").rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("._")
                ) if (patient_dir / "ADC").exists() else []
                dwi_paths = sorted(
                    p for p in (patient_dir / "DWI").rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("._")
                ) if (patient_dir / "DWI").exists() else []

                pair_count = min(len(adc_paths), len(dwi_paths))
                if pair_count == 0:
                    continue

                patient_key = f"{split}/{label}/{patient_dir.name}"
                by_label[label].append(
                    {
                        "patient_id": patient_key,
                        "label": LABEL_TO_ID[label],
                        "label_name": label,
                        "split": split,
                        "pairs": list(zip(adc_paths[:pair_count], dwi_paths[:pair_count])),
                    }
                )

        for label in labels:
            selected = by_label[label]
            if max_patients_per_class is not None:
                selected = selected[:max_patients_per_class]
            split_records.extend(selected)

        records.extend(split_records)

    return records


class MRIPairDataset(Dataset):
    def __init__(self, patient_records, transform):
        self.samples = []
        self.transform = transform
        for record in patient_records:
            for adc_path, dwi_path in record["pairs"]:
                self.samples.append(
                    {
                        "adc_path": adc_path,
                        "dwi_path": dwi_path,
                        "patient_id": record["patient_id"],
                        "label": record["label"],
                        "split": record["split"],
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
        label = torch.tensor(sample["label"], dtype=torch.float32)
        return image, label, sample["patient_id"]


def build_transforms(image_size):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


def split_records(records):
    return {
        split: [record for record in records if record["split"] == split]
        for split in SPLITS
    }


def patient_level_metrics(patient_logits, patient_labels, threshold=0.5):
    patient_ids = sorted(patient_logits)
    probs = np.array([torch.sigmoid(torch.tensor(np.mean(patient_logits[pid]))).item() for pid in patient_ids])
    labels = np.array([patient_labels[pid] for pid in patient_ids])
    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc": float("nan"),
        "num_patients": len(patient_ids),
        "num_co": int((labels == 1).sum()),
        "num_khong": int((labels == 0).sum()),
    }
    if len(np.unique(labels)) == 2:
        metrics["auc"] = roc_auc_score(labels, probs)
    return metrics, patient_ids, labels, probs, preds


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_count = 0
    patient_logits = defaultdict(list)
    patient_labels = {}

    with torch.no_grad():
        for images, labels, patient_ids in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            for logit, label, patient_id in zip(logits.detach().cpu(), labels.detach().cpu(), patient_ids):
                patient_logits[patient_id].append(float(logit))
                patient_labels[patient_id] = int(label.item())

    metrics, patient_ids, labels, probs, preds = patient_level_metrics(patient_logits, patient_labels, threshold)
    metrics["loss"] = float(total_loss / max(total_count, 1))
    return metrics, patient_ids, labels, probs, preds


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_count = 0

    for images, labels, _ in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        use_amp = device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


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


def format_metrics(metrics):
    return {key: round_float(value) if isinstance(value, (float, np.floating)) else value for key, value in metrics.items()}


def save_predictions(path, patient_ids, labels, probs, preds):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "true_label", "true_name", "prob_co", "pred_label", "pred_name"])
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


def save_checkpoint(path, model, epoch, val_metrics, best_model_metric):
    torch.save(
        {
            "model_state": unwrap_model(model).state_dict(),
            "model_name": "resnet50",
            "image_size": IMAGE_SIZE,
            "label_to_id": LABEL_TO_ID,
            "epoch": epoch,
            "val_metrics": rounded_metrics(val_metrics),
            "best_model_metric": best_model_metric,
            "input_channels": "[ADC, DWI, zeros]",
        },
        path,
    )


def plot_history(history, output_dir):
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(9, 4), dpi=150)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [row["val_acc"] for row in history], label="val acc")
    plt.plot(epochs, [row["val_f1"] for row in history], label="val f1")
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png")
    plt.close()


def main():
    seed_everything(SEED)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(FORCE_CPU)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    image_root = resolve_image_root(IMAGE_ROOT)
    records = collect_patient_pairs(
        image_root=image_root,
        splits=SPLITS,
        labels=LABELS,
        max_patients_per_class=MAX_PATIENTS_PER_CLASS,
    )
    records_by_split = split_records(records)

    train_records = records_by_split.get("train", [])
    val_records = records_by_split.get("val", [])
    test_records = records_by_split.get("test", [])
    if not train_records or not val_records:
        raise RuntimeError("Need non-empty train and val splits.")

    train_tf, eval_tf = build_transforms(IMAGE_SIZE)
    train_loader = DataLoader(
        MRIPairDataset(train_records, train_tf),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        MRIPairDataset(val_records, eval_tf),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        MRIPairDataset(test_records, eval_tf),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    ) if test_records else None

    model = build_model()
    model = wrap_model_for_device(model, device, USE_MULTI_GPU)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print(f"Image root: {image_root.resolve()}")
    print(f"Device: {device}")
    print(f"Train patients: {len(train_records)} | Val patients: {len(val_records)} | Test patients: {len(test_records)}")
    print(f"Train slices: {len(train_loader.dataset)} | Val slices: {len(val_loader.dataset)}")

    best_checkpoints = {
        "auc": {"score": -1.0, "path": output_dir / "best_auc_model.pt", "metric_key": "auc"},
    }
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_metrics, val_patient_ids, val_labels, val_probs, val_preds = evaluate(model, val_loader, device, THRESHOLD)

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

        for metric_name, tracker in best_checkpoints.items():
            score = val_metrics[tracker["metric_key"]]
            if score > tracker["score"]:
                tracker["score"] = score
                tracker["epoch"] = epoch
                save_checkpoint(tracker["path"], model, epoch, val_metrics, metric_name)
                save_predictions(
                    output_dir / f"val_predictions_best_{metric_name}.csv",
                    val_patient_ids,
                    val_labels,
                    val_probs,
                    val_preds,
                )

    with open(output_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, output_dir)

    checkpoint_metrics = {}
    for metric_name, tracker in best_checkpoints.items():
        checkpoint_name = f"best_{metric_name}"
        checkpoint = torch.load(tracker["path"], map_location=device, weights_only=False)
        unwrap_model(model).load_state_dict(checkpoint["model_state"])

        val_metrics, val_patient_ids, val_labels, val_probs, val_preds = evaluate(model, val_loader, device, THRESHOLD)
        save_predictions(
            output_dir / f"val_predictions_{checkpoint_name}.csv",
            val_patient_ids,
            val_labels,
            val_probs,
            val_preds,
        )
        checkpoint_metrics[checkpoint_name] = {"val": rounded_metrics(val_metrics)}

        print(f"Checkpoint {checkpoint_name} | epoch={checkpoint['epoch']} | val metrics: {format_metrics(val_metrics)}")
        print(f"Checkpoint {checkpoint_name} validation confusion matrix:")
        print(confusion_matrix(val_labels, val_preds))

        if test_loader is not None:
            test_metrics, test_patient_ids, test_labels, test_probs, test_preds = evaluate(model, test_loader, device, THRESHOLD)
            save_predictions(
                output_dir / f"test_predictions_{checkpoint_name}.csv",
                test_patient_ids,
                test_labels,
                test_probs,
                test_preds,
            )
            checkpoint_metrics[checkpoint_name]["test"] = rounded_metrics(test_metrics)
            print(f"Checkpoint {checkpoint_name} | test metrics: {format_metrics(test_metrics)}")
            print(f"Checkpoint {checkpoint_name} test confusion matrix:")
            print(confusion_matrix(test_labels, test_preds))

    with open(output_dir / "checkpoint_metrics.json", "w", encoding="utf-8") as f:
        json.dump(checkpoint_metrics, f, ensure_ascii=False, indent=2)

    for metric_name, tracker in best_checkpoints.items():
        print(f"Saved best_{metric_name} model: {tracker['path'].resolve()}")
    print(f"Saved outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
