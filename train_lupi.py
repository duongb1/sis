import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from sislib.common import get_device, resolve_max_len, round_float, round_metrics, seed_all, split_records, to_device, unwrap
from sislib.metrics import save_preds
from sislib.mri import collect_pairs
from sislib.mri_teacher import compute_mri_logits, load_mri_teacher, shuffle_teacher_for_train, split_teacher_stats
from sislib.text_data import TextDataset, collect_paired_text, save_records
from sislib.text_train import ce_epoch, eval_text


def parse_args():
    p = argparse.ArgumentParser(description="Paired-text LUPI training with MRI-guided CE sample weights.")
    p.add_argument("--images", default="/kaggle/input/datasets/duongb/cthsis/images")
    p.add_argument("--teacher", default="/kaggle/working/mri_classifier/best_auc_model.pt")
    p.add_argument("--student", default="/kaggle/working/text_phobert_classifier/best_auc_phobert")
    p.add_argument("--out", default="/kaggle/working/paired_text_lupi_from_mri")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch-text", type=int, default=16)
    p.add_argument("--batch-mri", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--alpha-lupi", type=float, default=0.2)
    p.add_argument("--weight-min", type=float, default=0.75)
    p.add_argument("--weight-max", type=float, default=1.25)
    p.add_argument("--shuffle-teacher", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    return p.parse_args()


def lupi_disagreement_loss(student_logits, teacher_logits, labels, alpha, weight_min, weight_max):
    ce = F.cross_entropy(student_logits, labels, reduction="none")
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_conf_gt = torch.gather(teacher_probs, dim=1, index=labels.unsqueeze(1)).squeeze(1)
        student_conf_gt = torch.gather(student_probs, dim=1, index=labels.unsqueeze(1)).squeeze(1)
        gap = torch.clamp(teacher_conf_gt - student_conf_gt, min=0.0)
        weights = 1.0 + alpha * gap
        weights = torch.clamp(weights, min=weight_min, max=weight_max)
    return (weights * ce).mean(), ce.mean().detach(), weights.mean().detach()


def lupi_epoch(model, loader, optimizer, scheduler, scaler, device, args):
    model.train()
    total_loss = total_ce = total_weight = 0.0
    total_count = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="Training LUPI", leave=False), start=1):
        batch.pop("id")
        labels = batch.pop("labels").to(device, non_blocking=True)
        teacher_logits = batch.pop("teacher_logits").to(device, non_blocking=True)
        inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(**inputs)
            loss, ce, weight = lupi_disagreement_loss(outputs.logits, teacher_logits, labels, args.alpha_lupi, args.weight_min, args.weight_max)
            loss = loss / args.accum
        scaler.scale(loss).backward()
        if step % args.accum == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        bs = labels.size(0)
        total_loss += loss.item() * args.accum * bs
        total_ce += ce.item() * bs
        total_weight += weight.item() * bs
        total_count += bs
    denom = max(total_count, 1)
    return float(total_loss / denom), float(total_ce / denom), float(total_weight / denom)


def save_best(model, tokenizer, path, epoch, metrics, args, max_len):
    path.mkdir(parents=True, exist_ok=True)
    unwrap(model).save_pretrained(path)
    tokenizer.save_pretrained(path)
    with open(path / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "epoch": epoch,
                "best_model_metric": "auc",
                "val_metrics": round_metrics(metrics),
                "student_init": args.student,
                "mri_teacher": args.teacher,
                "lupi_weighting": "teacher_student_confidence_gap",
                "alpha_lupi": args.alpha_lupi,
                "weight_min": args.weight_min,
                "weight_max": args.weight_max,
                "shuffle_teacher": args.shuffle_teacher,
                "max_length": max_len,
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
    multi_gpu = not args.no_mgpu

    text_records, missing, image_root = collect_paired_text(args.images)
    if missing:
        (out / "missing_text_patients.txt").write_text("\n".join(missing), encoding="utf-8")
    mri_records, _ = collect_pairs(args.images)
    train_rows, val_rows, test_rows = [split_records(text_records, s) for s in ("train", "val", "test")]
    print(f"Image root: {image_root.resolve()}")
    print(f"Paired text patients: {len(text_records)} | Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)}")

    use_lupi = args.alpha_lupi > 0
    teacher_logits = None
    if use_lupi:
        teacher = load_mri_teacher(args.teacher, device, multi_gpu)
        teacher_logits = compute_mri_logits(text_records, mri_records, teacher, device, args.batch_mri, args.workers)
        teacher_stats = split_teacher_stats(train_rows, val_rows, test_rows, teacher_logits, args.threshold)
        with open(out / "teacher_stats.json", "w", encoding="utf-8") as f:
            json.dump(teacher_stats, f, ensure_ascii=False, indent=2)
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()

        train_rows = [row for row in train_rows if row["id"] in teacher_logits]
        if args.shuffle_teacher:
            teacher_logits = shuffle_teacher_for_train(teacher_logits, train_rows, args.seed)
        print(f"Teacher train stats: {teacher_stats['train']}")
        print(f"Teacher val stats: {teacher_stats['val']}")
        print(f"Teacher test stats: {teacher_stats['test']}")
        print(f"LUPI train rows with MRI signal: {len(train_rows)}")
    else:
        print("alpha_lupi <= 0: running exact CE-only path without loading MRI teacher.")
    if not train_rows or not val_rows:
        raise RuntimeError("Need non-empty train and val records.")

    tokenizer = AutoTokenizer.from_pretrained(args.student, use_fast=False)
    student = AutoModelForSequenceClassification.from_pretrained(
        args.student,
        num_labels=2,
        id2label={0: "khong", 1: "co"},
        label2id={"khong": 0, "co": 1},
    )
    max_len = resolve_max_len(student, args.max_len)
    if max_len != args.max_len:
        print(f"Requested max_len={args.max_len}, using {max_len}.")
    student = to_device(student, device, multi_gpu)

    train_dataset = TextDataset(train_rows, tokenizer, max_len, teacher_logits if use_lupi else None)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_text,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(TextDataset(val_rows, tokenizer, max_len), batch_size=args.batch_text, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(TextDataset(test_rows, tokenizer, max_len), batch_size=args.batch_text, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda") if test_rows else None

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, int(np.ceil(len(train_loader) / max(args.accum, 1))) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * args.warmup), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_name = "best_auc_lupi" if use_lupi else "best_auc_ce"
    best, best_dir, history = -1.0, out / best_name, []
    for epoch in range(1, args.epochs + 1):
        if use_lupi:
            train_loss, train_ce, train_weight = lupi_epoch(student, train_loader, optimizer, scheduler, scaler, device, args)
        else:
            train_loss = ce_epoch(student, train_loader, optimizer, scheduler, scaler, device, args.accum)
            train_ce = train_loss
            train_weight = 1.0
        val_metrics, val_ids, val_y, val_p, val_pred = eval_text(student, val_loader, device, args.threshold)
        row = {
            "epoch": epoch,
            "train_loss": round_float(train_loss),
            "train_ce": round_float(train_ce),
            "train_weight": round_float(train_weight),
            "val_loss": round_float(val_metrics["loss"]),
            "val_acc": round_float(val_metrics["accuracy"]),
            "val_f1": round_float(val_metrics["f1"]),
            "val_auc": round_float(val_metrics["auc"]),
        }
        history.append(row)
        mode_name = "LUPI" if use_lupi else "CE"
        print(f"Epoch {epoch:03d}/{args.epochs} [{mode_name}] | train_loss={train_loss:.3f} | val_f1={val_metrics['f1']:.3f} | val_auc={val_metrics['auc']:.3f}")
        if val_metrics["auc"] > best:
            best = val_metrics["auc"]
            save_best(student, tokenizer, best_dir, epoch, val_metrics, args, max_len)
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
