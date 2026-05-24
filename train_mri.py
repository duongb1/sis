import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from sislib.common import get_device, round_float, round_metrics, seed_all, split_records, to_device, unwrap
from sislib.metrics import cls_metrics, save_preds
from sislib.mri import MRIDataset, collect_pairs, mri_transforms, resnet50_binary


def parse_args():
    p = argparse.ArgumentParser(description="Train MRI-only ResNet50 classifier.")
    p.add_argument("--images", default="/kaggle/input/datasets/duongbui/siscth")
    p.add_argument("--out", default="/kaggle/working/mri_classifier")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    return p.parse_args()


def train_epoch(model, loader, opt, scaler, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = total_count = 0
    for images, labels, _ in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_count += bs
    return total_loss / max(total_count, 1)


@torch.no_grad()
def eval_mri(model, loader, device, threshold):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = total_count = 0
    logits_by_id, labels_by_id = defaultdict(list), {}
    for images, labels, ids in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_count += bs
        for logit, label, item_id in zip(logits.detach().cpu(), labels.detach().cpu(), ids):
            logits_by_id[item_id].append(float(logit))
            labels_by_id[item_id] = int(label.item())
    ids = sorted(logits_by_id)
    patient_logits = np.array([np.mean(logits_by_id[i]) for i in ids], dtype=np.float32)
    probs = np.array([torch.sigmoid(torch.tensor(logit)).item() for logit in patient_logits])
    labels = np.array([labels_by_id[i] for i in ids])
    preds = (probs >= threshold).astype(np.int64)
    metrics = cls_metrics(labels, probs, preds, loss=total_loss / max(total_count, 1), id_name="num_patients", threshold=threshold)
    return metrics, ids, labels, patient_logits, probs, preds


def save_mri_teacher_outputs(path, ids, labels, logits, probs, preds):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "true_label", "logit_co", "prob_co", "pred_label"])
        writer.writeheader()
        for item_id, label, logit, prob, pred in zip(ids, labels, logits, probs, preds):
            writer.writerow(
                {
                    "id": item_id,
                    "true_label": int(label),
                    "logit_co": round_float(logit),
                    "prob_co": round_float(prob),
                    "pred_label": int(pred),
                }
            )


def save_ckpt(path, model, epoch, metrics, args):
    torch.save(
        {
            "model_state": unwrap(model).state_dict(),
            "model_name": "resnet50",
            "image_size": args.size,
            "epoch": epoch,
            "val_metrics": round_metrics(metrics),
            "best_model_metric": "auc",
            "input_channels": "[ADC, DWI, zeros]",
        },
        path,
    )


def main():
    args = parse_args()
    seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.cpu)

    records, image_root = collect_pairs(args.images, args.max_per_class)
    train_rows, val_rows, test_rows = [split_records(records, s) for s in ("train", "val", "test")]
    if not train_rows or not val_rows:
        raise RuntimeError("Need non-empty train and val records.")
    print(f"Image root: {image_root.resolve()}")
    print(f"Patients train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    train_tf, eval_tf = mri_transforms(args.size)
    train_loader = DataLoader(MRIDataset(train_rows, train_tf), batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(MRIDataset(val_rows, eval_tf), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(MRIDataset(test_rows, eval_tf), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda") if test_rows else None

    model = to_device(resnet50_binary(), device, not args.no_mgpu)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best, best_path, history = -1.0, out / "best_auc_model.pt", []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, opt, scaler, device)
        val_metrics, val_ids, val_y, val_logits, val_p, val_pred = eval_mri(model, val_loader, device, args.threshold)
        history.append({"epoch": epoch, "train_loss": round_float(train_loss), "val_loss": round_float(val_metrics["loss"]), "val_acc": round_float(val_metrics["accuracy"]), "val_f1": round_float(val_metrics["f1"]), "val_auc": round_float(val_metrics["auc"])})
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.3f} | val_auc={val_metrics['auc']:.3f}")
        if val_metrics["auc"] > best:
            best = val_metrics["auc"]
            save_ckpt(best_path, model, epoch, val_metrics, args)
            save_preds(out / "val_predictions_best_auc.csv", val_ids, val_y, val_p, val_pred, "id")
            save_mri_teacher_outputs(out / "val_teacher_outputs_best_auc.csv", val_ids, val_y, val_logits, val_p, val_pred)

    with open(out / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    unwrap(model).load_state_dict(ckpt["model_state"])
    all_metrics = {}
    for split, loader in [("val", val_loader), ("test", test_loader)]:
        if loader is None:
            continue
        metrics, ids, y, logits, p, pred = eval_mri(model, loader, device, args.threshold)
        all_metrics[split] = round_metrics(metrics)
        save_preds(out / f"{split}_predictions_best_auc.csv", ids, y, p, pred, "id")
        save_mri_teacher_outputs(out / f"{split}_teacher_outputs_best_auc.csv", ids, y, logits, p, pred)
        print(f"{split}: {round_metrics(metrics)}")
        print(confusion_matrix(y, pred, labels=[0, 1]))
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
