import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from sislib.common import batch_to_device, get_device, resolve_max_len, round_float, round_metrics, seed_all, split_records, to_device, unwrap
from sislib.metrics import save_preds
from sislib.mri import collect_pairs
from sislib.mri_teacher import compute_mri_logits, load_mri_teacher, split_teacher_stats
from sislib.text_data import TextDataset, collect_paired_text, save_records
from sislib.text_train import eval_text


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune large-text PhoBERT with MRI-guided hard-negative sample weights.")
    p.add_argument("--student", required=True, help="Large-text checkpoint used as student initialization.")
    p.add_argument("--teacher", required=True, help="MRI-only teacher checkpoint.")
    p.add_argument("--images", default="/kaggle/input/datasets/duongb/cthsis/images")
    p.add_argument("--out", default="/kaggle/working/hard_negative_reweight")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch-text", type=int, default=16)
    p.add_argument("--batch-mri", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--best-metric", choices=["auc", "f1", "specificity"], default="auc")
    p.add_argument("--text-fp-threshold", type=float, default=0.7)
    p.add_argument("--mri-negative-threshold", type=float, default=0.3)
    p.add_argument("--mri-ambiguous-threshold", type=float, default=0.5)
    p.add_argument("--hard-negative-weight", type=float, default=3.0)
    p.add_argument("--ambiguous-negative-weight", type=float, default=1.0)
    p.add_argument("--positive-weight", type=float, default=1.0)
    p.add_argument("--normal-weight", type=float, default=1.0)
    p.add_argument("--shuffle-teacher", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    return p.parse_args()


def save_model(model, tokenizer, path, epoch, metrics, args, max_len, weight_stats):
    path.mkdir(parents=True, exist_ok=True)
    unwrap(model).save_pretrained(path)
    tokenizer.save_pretrained(path)
    with open(path / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "epoch": epoch,
                "best_model_metric": args.best_metric,
                "val_metrics": round_metrics(metrics),
                "student_init": args.student,
                "mri_teacher": args.teacher,
                "max_length": max_len,
                "weight_stats": weight_stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


@torch.no_grad()
def compute_text_probs(records, model, tokenizer, max_len, device, batch_size, workers):
    loader = DataLoader(
        TextDataset(records, tokenizer, max_len),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    out = {}
    for batch in tqdm(loader, desc="Computing large-text probabilities", leave=False):
        inputs, ids = batch_to_device(batch, device, "id")
        inputs.pop("labels")
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        for item_id, prob in zip(ids, probs):
            out[item_id] = float(prob)
    return out


def prob_co_from_logits(logits):
    values = np.asarray(logits, dtype=np.float32)
    values = values - values.max()
    exp_values = np.exp(values)
    return float(exp_values[1] / exp_values.sum())


def maybe_shuffle_probs(probs, rows, seed):
    rng = random.Random(seed)
    ids = [row["id"] for row in rows if row["id"] in probs]
    shuffled = ids[:]
    rng.shuffle(shuffled)
    out = dict(probs)
    for target_id, source_id in zip(ids, shuffled):
        out[target_id] = probs[source_id]
    return out


def build_sample_weights(rows, text_probs, mri_probs, args):
    weights, details = {}, []
    counts = {"hard_negative": 0, "ambiguous_negative": 0, "positive": 0, "normal": 0, "missing_signal": 0}
    for row in rows:
        item_id = row["id"]
        label = row["label"]
        p_text = text_probs.get(item_id)
        p_mri = mri_probs.get(item_id)
        rule = "normal"
        weight = args.normal_weight
        if label == 1 and args.positive_weight != args.normal_weight:
            rule = "positive"
            weight = args.positive_weight
        elif p_text is None or p_mri is None:
            rule = "missing_signal"
        elif label == 0 and p_text >= args.text_fp_threshold and p_mri <= args.mri_negative_threshold:
            rule = "hard_negative"
            weight = args.hard_negative_weight
        elif label == 0 and p_text >= args.text_fp_threshold and p_mri > args.mri_ambiguous_threshold:
            rule = "ambiguous_negative"
            weight = args.ambiguous_negative_weight
        counts[rule] += 1
        weights[item_id] = float(weight)
        details.append(
            {
                "id": item_id,
                "label": label,
                "p_text_co": "" if p_text is None else round_float(p_text),
                "p_mri_co": "" if p_mri is None else round_float(p_mri),
                "sample_weight": round_float(weight),
                "rule": rule,
            }
        )
    return weights, details, counts


def weighted_ce_epoch(model, loader, optimizer, scheduler, scaler, device, accum):
    model.train()
    total_loss = total_ce = total_weight = 0.0
    total_count = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="Training hard-negative reweight", leave=False), start=1):
        inputs, _ = batch_to_device(batch, device, "id")
        labels = inputs.pop("labels")
        weights = inputs.pop("sample_weight")
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(**inputs).logits
            ce = F.cross_entropy(logits, labels, reduction="none")
            loss = (weights * ce).mean()
            scaled_loss = loss / accum
        scaler.scale(scaled_loss).backward()
        if step % accum == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_ce += ce.mean().item() * bs
        total_weight += weights.mean().item() * bs
        total_count += bs
    denom = max(total_count, 1)
    return float(total_loss / denom), float(total_ce / denom), float(total_weight / denom)


def main():
    args = parse_args()
    seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    text_records, missing, image_root = collect_paired_text(args.images)
    if missing:
        (out / "missing_text_files.txt").write_text("\n".join(missing), encoding="utf-8")
    mri_records, _ = collect_pairs(args.images)
    train_rows, val_rows, test_rows = [split_records(text_records, s) for s in ("train", "val", "test")]
    if not train_rows or not val_rows:
        raise RuntimeError("Need non-empty paired train and val records.")

    device = get_device(args.cpu)
    tokenizer = AutoTokenizer.from_pretrained(args.student, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.student,
        num_labels=2,
        id2label={0: "khong", 1: "co"},
        label2id={"khong": 0, "co": 1},
    )
    max_len = resolve_max_len(model, args.max_len)
    if max_len != args.max_len:
        print(f"Requested max_len={args.max_len}, using {max_len}.")
    model = to_device(model, device, not args.no_mgpu)

    teacher = load_mri_teacher(args.teacher, device, not args.no_mgpu)
    teacher_logits = compute_mri_logits(train_rows + val_rows + test_rows, mri_records, teacher, device, args.batch_mri, args.workers)
    teacher_stats = split_teacher_stats(train_rows, val_rows, test_rows, teacher_logits, args.threshold)
    with open(out / "teacher_stats.json", "w", encoding="utf-8") as f:
        json.dump(teacher_stats, f, ensure_ascii=False, indent=2)
    print(f"Teacher train stats: {teacher_stats['train']}")
    print(f"Teacher val stats: {teacher_stats['val']}")
    print(f"Teacher test stats: {teacher_stats['test']}")

    text_probs = compute_text_probs(train_rows, model, tokenizer, max_len, device, args.batch_text, args.workers)
    mri_probs = {item_id: prob_co_from_logits(logits) for item_id, logits in teacher_logits.items()}
    if args.shuffle_teacher:
        mri_probs = maybe_shuffle_probs(mri_probs, train_rows, args.seed)

    sample_weights, weight_details, weight_counts = build_sample_weights(train_rows, text_probs, mri_probs, args)
    weight_stats = {
        "counts": weight_counts,
        "mean_weight": round_float(np.mean(list(sample_weights.values()))),
        "shuffle_teacher": bool(args.shuffle_teacher),
    }
    print(f"Weight stats: {weight_stats}")
    with open(out / "sample_weights.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "p_text_co", "p_mri_co", "sample_weight", "rule"])
        writer.writeheader()
        writer.writerows(weight_details)

    train_loader = DataLoader(
        TextDataset(train_rows, tokenizer, max_len, sample_weights=sample_weights),
        batch_size=args.batch_text,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(TextDataset(val_rows, tokenizer, max_len), batch_size=args.batch_text, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(TextDataset(test_rows, tokenizer, max_len), batch_size=args.batch_text, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, int(np.ceil(len(train_loader) / max(args.accum, 1))) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * args.warmup), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best, best_dir, history = -1.0, out / "best_auc_phobert", []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_ce, train_weight = weighted_ce_epoch(model, train_loader, optimizer, scheduler, scaler, device, args.accum)
        val_metrics, val_ids, val_y, val_p, val_pred = eval_text(model, val_loader, device, args.threshold)
        score = val_metrics[args.best_metric]
        history.append(
            {
                "epoch": epoch,
                "train_loss": round_float(train_loss),
                "train_ce": round_float(train_ce),
                "train_weight": round_float(train_weight),
                "val_loss": round_float(val_metrics["loss"]),
                "val_acc": round_float(val_metrics["accuracy"]),
                "val_f1": round_float(val_metrics["f1"]),
                "val_auc": round_float(val_metrics["auc"]),
                "val_sensitivity": round_float(val_metrics["sensitivity"]),
                "val_specificity": round_float(val_metrics["specificity"]),
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.3f} | "
            f"train_ce={train_ce:.3f} | weight={train_weight:.3f} | "
            f"val_acc={val_metrics['accuracy']:.3f} | val_f1={val_metrics['f1']:.3f} | "
            f"val_auc={val_metrics['auc']:.3f} | val_spec={val_metrics['specificity']:.3f}"
        )
        if score > best:
            best = score
            save_model(model, tokenizer, best_dir, epoch, val_metrics, args, max_len, weight_stats)
            save_preds(out / "val_predictions_best_auc.csv", val_ids, val_y, val_p, val_pred, "id")

    with open(out / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    save_records(out / "dataset_records.csv", text_records)

    eval_model = to_device(AutoModelForSequenceClassification.from_pretrained(best_dir), device, not args.no_mgpu)
    all_metrics = {}
    for split, rows in [("val", val_rows), ("test", test_rows)]:
        loader = DataLoader(TextDataset(rows, tokenizer, max_len), batch_size=args.batch_text, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
        metrics, ids, y, p, pred = eval_text(eval_model, loader, device, args.threshold, desc=f"Evaluating {split}")
        all_metrics[split] = round_metrics(metrics)
        save_preds(out / f"{split}_predictions_best_auc.csv", ids, y, p, pred, "id")
        print(f"{split}: {round_metrics(metrics)}")
        print(confusion_matrix(y, pred, labels=[0, 1]))
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
