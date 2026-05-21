import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from sislib.common import get_device, round_float, round_metrics, seed_all, split_records, to_device, unwrap
from sislib.metrics import save_preds
from sislib.mri import MRIDataset, collect_pairs, resnet50_binary
from sislib.text_data import TextDataset, collect_paired_text, save_records
from sislib.text_train import eval_text, kd_epoch, kd_loss_fn


def parse_args():
    p = argparse.ArgumentParser(description="MRI teacher -> paired-text student KD.")
    p.add_argument("--images", default="/kaggle/input/datasets/duongb/cthsis/images")
    p.add_argument("--teacher", default="/kaggle/working/mri_classifier/best_auc_model.pt")
    p.add_argument("--student", default="/kaggle/working/text_phobert_classifier/best_auc_phobert")
    p.add_argument("--out", default="/kaggle/working/text_kd_from_correct_mri_teacher")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch-text", type=int, default=16)
    p.add_argument("--batch-mri", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--kd", choices=["binary", "kl"], default="binary")
    p.add_argument("--temp", type=float, default=2.0)
    p.add_argument("--ce-warmup", type=int, default=0)
    p.add_argument("--teacher-correct-only", action="store_true", default=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    return p.parse_args()


def clean_state_dict(state):
    return {k.replace("module.", "", 1): v for k, v in state.items()} if any(k.startswith("module.") for k in state) else state


def load_mri(path, device, multi_gpu):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model = resnet50_binary()
    model.load_state_dict(clean_state_dict(state))
    return to_device(model.eval(), device, multi_gpu)


@torch.no_grad()
def mri_teacher_logits(text_records, mri_records, teacher, device, batch_size, workers, threshold):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(MRIDataset(mri_records, tf), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    by_id, labels_by_id = {}, {}
    for images, labels, ids in tqdm(loader, desc="Computing MRI teacher logits"):
        logits = teacher(images.to(device, non_blocking=True)).squeeze(1).detach().cpu().numpy()
        for logit, label, item_id in zip(logits, labels, ids):
            by_id.setdefault(item_id, []).append(float(logit))
            labels_by_id[item_id] = int(label)

    text_ids = {r["id"] for r in text_records}
    logits_2, correct = {}, {}
    labels, probs = [], []
    for item_id, vals in by_id.items():
        if item_id not in text_ids:
            continue
        z = float(np.mean(vals))
        p = 1.0 / (1.0 + np.exp(-z))
        y = labels_by_id[item_id]
        logits_2[item_id] = [-z / 2.0, z / 2.0]
        correct[item_id] = int(p >= threshold) == y
        labels.append(y)
        probs.append(p)
    labels = np.array(labels, dtype=np.int64)
    probs = np.array(probs, dtype=np.float32)
    preds = (probs >= threshold).astype(np.int64)
    stats = {
        "teacher_accuracy": float(accuracy_score(labels, preds)),
        "teacher_f1": float(f1_score(labels, preds, zero_division=0)),
        "teacher_auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else float("nan"),
        "teacher_num_patients": int(len(labels)),
        "teacher_correct_patients": int((preds == labels).sum()),
    }
    return logits_2, correct, stats


def main():
    args = parse_args()
    seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.cpu)
    multi_gpu = not args.no_mgpu

    text_records, missing, image_root = collect_paired_text(args.images)
    if missing:
        (out / "missing_text_patients.txt").write_text("\n".join(missing), encoding="utf-8")
    mri_records, _ = collect_pairs(args.images)
    train_rows, val_rows, test_rows = [split_records(text_records, s) for s in ("train", "val", "test")]
    print(f"Image root: {image_root.resolve()}")
    print(f"Text patients: {len(text_records)} | Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)}")

    teacher = load_mri(args.teacher, device, multi_gpu)
    logits, correct, stats = mri_teacher_logits(text_records, mri_records, teacher, device, args.batch_mri, args.workers, args.threshold)
    if args.teacher_correct_only:
        logits = {k: v for k, v in logits.items() if correct.get(k, False)}
        train_rows = [r for r in train_rows if r["id"] in logits]
    print(f"Teacher stats: {round_metrics(stats)} | Train rows after filter: {len(train_rows)}")
    with open(out / "teacher_stats.json", "w", encoding="utf-8") as f:
        json.dump(round_metrics(stats), f, ensure_ascii=False, indent=2)
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained(args.student, use_fast=False)
    student = AutoModelForSequenceClassification.from_pretrained(args.student, num_labels=2, id2label={0: "khong", 1: "co"}, label2id={"khong": 0, "co": 1})
    max_len = min(args.max_len, getattr(student.config, "max_position_embeddings", args.max_len + 2) - 2)
    student = to_device(student, device, multi_gpu)

    train_loader = DataLoader(TextDataset(train_rows, tok, max_len, logits), batch_size=args.batch_text, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(TextDataset(val_rows, tok, max_len), batch_size=args.batch_text, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(TextDataset(test_rows, tok, max_len), batch_size=args.batch_text, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda") if test_rows else None

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, int(np.ceil(len(train_loader) / max(args.accum, 1))) * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(steps * args.warmup), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = kd_loss_fn(args.alpha, args.kd, args.temp, args.ce_warmup)
    best, best_dir, history = -1.0, out / "best_auc_mri_text_kd", []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ce, tr_kd = kd_epoch(student, train_loader, opt, sched, scaler, device, loss_fn, epoch, args.accum)
        val_metrics, val_ids, val_y, val_p, val_pred = eval_text(student, val_loader, device, args.threshold)
        history.append({"epoch": epoch, "train_loss": round_float(tr_loss), "train_ce": round_float(tr_ce), "train_kd": round_float(tr_kd), "val_loss": round_float(val_metrics["loss"]), "val_acc": round_float(val_metrics["accuracy"]), "val_f1": round_float(val_metrics["f1"]), "val_auc": round_float(val_metrics["auc"])})
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={tr_loss:.3f} | val_auc={val_metrics['auc']:.3f}")
        if val_metrics["auc"] > best:
            best = val_metrics["auc"]
            best_dir.mkdir(parents=True, exist_ok=True)
            unwrap(student).save_pretrained(best_dir)
            tok.save_pretrained(best_dir)
            save_preds(out / "val_predictions_best_auc.csv", val_ids, val_y, val_p, val_pred, "id")
    with open(out / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    save_records(out / "dataset_records.csv", text_records)

    eval_model = to_device(AutoModelForSequenceClassification.from_pretrained(best_dir), device, multi_gpu)
    all_metrics = {}
    for split, loader in [("val", val_loader), ("test", test_loader)]:
        if loader is None:
            continue
        metrics, ids, y, p, pred = eval_text(eval_model, loader, device, args.threshold)
        all_metrics[split] = round_metrics(metrics)
        save_preds(out / f"{split}_predictions_best_auc.csv", ids, y, p, pred, "id")
        print(f"{split}: {round_metrics(metrics)}")
        print(confusion_matrix(y, pred, labels=[0, 1]))
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
