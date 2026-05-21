import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from sislib.common import get_device, resolve_max_len, round_float, round_metrics, seed_all, split_records, to_device, unwrap
from sislib.metrics import save_preds
from sislib.mri import collect_pairs
from sislib.mri_teacher import compute_mri_logits, load_mri_teacher, shuffle_teacher_for_train, split_teacher_stats
from sislib.text_data import TextDataset, collect_paired_text, save_records
from sislib.text_train import eval_text, kd_epoch, kd_loss_fn


def parse_args():
    p = argparse.ArgumentParser(description="MRI teacher -> paired-text student KD.")
    p.add_argument("--images", default="/kaggle/input/datasets/duongb/cthsis/images")
    p.add_argument("--teacher", default="/kaggle/working/mri_classifier/best_auc_model.pt")
    p.add_argument("--student", default="/kaggle/working/text_phobert_classifier/best_auc_phobert")
    p.add_argument("--out", default="/kaggle/working/paired_text_kd_from_mri_teacher")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch-text", type=int, default=16)
    p.add_argument("--batch-mri", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--kd", choices=["binary", "kl"], default="binary")
    p.add_argument("--kd-weight", choices=["none", "confidence"], default="none")
    p.add_argument("--temp", type=float, default=2.0)
    p.add_argument("--ce-warmup", type=int, default=0)
    p.add_argument("--shuffle-teacher", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    return p.parse_args()


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

    teacher = load_mri_teacher(args.teacher, device, multi_gpu)
    logits = compute_mri_logits(text_records, mri_records, teacher, device, args.batch_mri, args.workers)
    stats = split_teacher_stats(train_rows, val_rows, test_rows, logits, args.threshold)
    train_rows = [r for r in train_rows if r["id"] in logits]
    if args.shuffle_teacher:
        logits = shuffle_teacher_for_train(logits, train_rows, args.seed)
    if not train_rows or not val_rows:
        raise RuntimeError("Need non-empty train and val records.")
    print(f"Teacher train stats: {stats['train']}")
    print(f"Teacher val stats: {stats['val']}")
    print(f"Teacher test stats: {stats['test']}")
    print(f"KD train rows with MRI signal: {len(train_rows)}")
    with open(out / "teacher_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained(args.student, use_fast=False)
    student = AutoModelForSequenceClassification.from_pretrained(args.student, num_labels=2, id2label={0: "khong", 1: "co"}, label2id={"khong": 0, "co": 1})
    max_len = resolve_max_len(student, args.max_len)
    if max_len != args.max_len:
        print(f"Requested max_len={args.max_len}, using {max_len}.")
    student = to_device(student, device, multi_gpu)

    train_loader = DataLoader(TextDataset(train_rows, tok, max_len, logits), batch_size=args.batch_text, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(TextDataset(val_rows, tok, max_len), batch_size=args.batch_text, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(TextDataset(test_rows, tok, max_len), batch_size=args.batch_text, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda") if test_rows else None

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, int(np.ceil(len(train_loader) / max(args.accum, 1))) * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(steps * args.warmup), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = kd_loss_fn(args.alpha, args.kd, args.temp, args.ce_warmup, args.kd_weight)
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
            with open(best_dir / "training_info.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "best_model_metric": "auc",
                        "val_metrics": round_metrics(val_metrics),
                        "student_init": args.student,
                        "mri_teacher": args.teacher,
                        "alpha": args.alpha,
                        "kd_loss": args.kd,
                        "kd_weight": args.kd_weight,
                        "temperature": args.temp,
                        "shuffle_teacher": args.shuffle_teacher,
                        "max_length": max_len,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
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
