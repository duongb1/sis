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
from sislib.text_data import TextDataset, collect_paired_text, save_records
from sislib.text_train import eval_text, kd_epoch, kd_loss_fn


def parse_args():
    p = argparse.ArgumentParser(description="Large-text teacher -> paired-text student KD.")
    p.add_argument("--images", default="/kaggle/input/datasets/duongb/cthsis/images")
    p.add_argument("--teacher", default="/kaggle/working/text_phobert_classifier/best_auc_phobert")
    p.add_argument("--student", default="vinai/phobert-base")
    p.add_argument("--out", default="/kaggle/working/paired_text_kd_from_large_text_teacher")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--kd", choices=["binary", "kl"], default="binary")
    p.add_argument("--temp", type=float, default=2.0)
    p.add_argument("--ce-warmup", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    return p.parse_args()


@torch.no_grad()
def teacher_logits(records, model, tokenizer, max_len, device, batch_size, workers):
    loader = DataLoader(TextDataset(records, tokenizer, max_len), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    model.eval()
    logits, labels_all, probs_all = {}, [], []
    for batch in loader:
        ids = batch.pop("id")
        labels = batch.pop("labels").to(device)
        inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(**inputs).logits.detach().cpu()
        probs = torch.softmax(out, dim=-1)[:, 1].numpy()
        for item_id, logit in zip(ids, out):
            logits[item_id] = logit.tolist()
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        probs_all.extend(probs.tolist())
    labels = np.array(labels_all, dtype=np.int64)
    probs = np.array(probs_all, dtype=np.float32)
    preds = (probs >= 0.5).astype(np.int64)
    from sislib.metrics import cls_metrics
    return logits, cls_metrics(labels, probs, preds, id_name="num_patients")


def save_model(model, tokenizer, path, epoch, val_metrics, args, max_len):
    path.mkdir(parents=True, exist_ok=True)
    unwrap(model).save_pretrained(path)
    tokenizer.save_pretrained(path)
    with open(path / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "epoch": epoch,
                "best_model_metric": "auc",
                "val_metrics": round_metrics(val_metrics),
                "teacher": args.teacher,
                "student": args.student,
                "max_length": max_len,
                "alpha": args.alpha,
                "kd_loss": args.kd,
                "temperature": args.temp,
                "temperature_used": args.kd == "kl",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    args = parse_args()
    seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.cpu)

    records, missing, image_root = collect_paired_text(args.images)
    if missing:
        (out / "missing_text_patients.txt").write_text("\n".join(missing), encoding="utf-8")
    train_rows, val_rows, test_rows = [split_records(records, s) for s in ("train", "val", "test")]
    print(f"Image root: {image_root.resolve()}")
    print(f"Paired text patients: {len(records)} | Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)}")

    t_tok = AutoTokenizer.from_pretrained(args.teacher, use_fast=False)
    teacher = to_device(AutoModelForSequenceClassification.from_pretrained(args.teacher), device, not args.no_mgpu)
    t_len = resolve_max_len(unwrap(teacher), args.max_len)
    t_logits, t_stats = teacher_logits(records, teacher, t_tok, t_len, device, args.batch, args.workers)
    print(f"Teacher stats: {round_metrics(t_stats)}")
    with open(out / "teacher_stats.json", "w", encoding="utf-8") as f:
        json.dump(round_metrics(t_stats), f, ensure_ascii=False, indent=2)
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    s_tok = AutoTokenizer.from_pretrained(args.student, use_fast=False)
    student = AutoModelForSequenceClassification.from_pretrained(args.student, num_labels=2, id2label={0: "khong", 1: "co"}, label2id={"khong": 0, "co": 1})
    s_len = resolve_max_len(student, args.max_len)
    student = to_device(student, device, not args.no_mgpu)

    train_loader = DataLoader(TextDataset(train_rows, s_tok, s_len, t_logits), batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(TextDataset(val_rows, s_tok, s_len, t_logits), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(TextDataset(test_rows, s_tok, s_len, t_logits), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda") if test_rows else None

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, int(np.ceil(len(train_loader) / max(args.accum, 1))) * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(steps * args.warmup), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = kd_loss_fn(args.alpha, args.kd, args.temp, args.ce_warmup)

    best, best_dir, history = -1.0, out / "best_auc_paired_text_kd", []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ce, tr_kd = kd_epoch(student, train_loader, opt, sched, scaler, device, loss_fn, epoch, args.accum)
        val_metrics, val_ids, val_y, val_p, val_pred = eval_text(student, val_loader, device, args.threshold)
        history.append({"epoch": epoch, "train_loss": round_float(tr_loss), "train_ce": round_float(tr_ce), "train_kd": round_float(tr_kd), "val_loss": round_float(val_metrics["loss"]), "val_acc": round_float(val_metrics["accuracy"]), "val_f1": round_float(val_metrics["f1"]), "val_auc": round_float(val_metrics["auc"])})
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={tr_loss:.3f} | ce={tr_ce:.3f} | kd={tr_kd:.3f} | val_auc={val_metrics['auc']:.3f}")
        if val_metrics["auc"] > best:
            best = val_metrics["auc"]
            save_model(student, s_tok, best_dir, epoch, val_metrics, args, s_len)
            save_preds(out / "val_predictions_best_auc.csv", val_ids, val_y, val_p, val_pred, "id")

    with open(out / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    save_records(out / "dataset_records.csv", records)

    eval_model = to_device(AutoModelForSequenceClassification.from_pretrained(best_dir), device, not args.no_mgpu)
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
