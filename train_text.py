import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sislib.common import quiet_hf_logging

quiet_hf_logging()

from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from sislib.common import get_device, resolve_max_len, round_float, round_metrics, seed_all, split_records, to_device, unwrap
from sislib.metrics import format_metrics_summary, save_preds
from sislib.text_data import (
    TextDataset,
    collect_excel_text,
    collect_large_text,
    discover_excel_labels,
    discover_text_labels,
    is_excel_data,
    make_label_maps,
    parse_labels_arg,
    save_records,
)
from sislib.text_train import ce_epoch, eval_text


def parse_args():
    p = argparse.ArgumentParser(description="Train text-only PhoBERT on large text files.")
    p.add_argument("--data", default="/kaggle/input/datasets/duongb/cthsis/data/texts")
    p.add_argument("--out", default="/kaggle/working/text_phobert_classifier")
    p.add_argument("--model", default="vinai/phobert-base")
    p.add_argument("--format", choices=["auto", "text", "excel"], default="auto", help="Input format. Excel accepts one .xlsx file, comma-separated .xlsx files, or a folder of .xlsx files.")
    p.add_argument("--excel-task", choices=["multiclass", "binary"], default="multiclass", help="For Excel input, train on LABEL multi-class targets or co/khong targets inferred from filenames.")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio for unsplit Excel input.")
    p.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio for unsplit Excel input.")
    p.add_argument("--split-strategy", choices=["random", "kfold"], default="random", help="Excel split strategy. kfold uses one fold as 20% test when --n-folds 5.")
    p.add_argument("--n-folds", type=int, default=5, help="Number of folds for --split-strategy kfold.")
    p.add_argument("--fold-index", type=int, default=0, help="Zero-based held-out test fold for --split-strategy kfold.")
    p.add_argument("--labels", default=None, help="Comma-separated class names. Defaults to CSV/file labels discovered under --data.")
    p.add_argument("--binary-positive-label", default=None, help="Class treated as positive for one-vs-rest binary metrics. Defaults to I63_INFARCTION, or co for --excel-task binary.")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    return p.parse_args()


def save_model(model, tokenizer, path, epoch, metrics, args, max_len, labels, label_to_id):
    path.mkdir(parents=True, exist_ok=True)
    unwrap(model).save_pretrained(path)
    tokenizer.save_pretrained(path)
    with open(path / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "epoch": epoch,
                "best_model_metric": "auc",
                "val_metrics": round_metrics(metrics),
                "text_model_name": args.model,
                "max_length": max_len,
                "num_labels": len(labels),
                "labels": labels,
                "label_to_id": label_to_id,
                "id_to_label": {str(index): label for index, label in enumerate(labels)},
                "binary_positive_label": args.binary_positive_label,
                "split_strategy": args.split_strategy,
                "n_folds": args.n_folds,
                "fold_index": args.fold_index,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
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

    use_excel = args.format == "excel" or (args.format == "auto" and is_excel_data(args.data))
    if use_excel:
        labels = parse_labels_arg(args.labels) or discover_excel_labels(args.data, task=args.excel_task)
        if args.binary_positive_label is None:
            args.binary_positive_label = "co" if args.excel_task == "binary" else "I63_INFARCTION"
    else:
        labels = parse_labels_arg(args.labels) or discover_text_labels(args.data)
        if args.binary_positive_label is None:
            args.binary_positive_label = "I63_INFARCTION"
    if len(labels) < 2:
        raise RuntimeError(f"Need at least two labels under --data, found: {labels}")
    label_to_id, id_to_label = make_label_maps(labels)

    if use_excel:
        records, skipped, data_root = collect_excel_text(
            args.data,
            labels=labels,
            label_to_id=label_to_id,
            task=args.excel_task,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_strategy=args.split_strategy,
            n_folds=args.n_folds,
            fold_index=args.fold_index,
        )
    else:
        records, skipped, data_root = collect_large_text(args.data, labels=labels, label_to_id=label_to_id)
    if skipped:
        (out / "skipped_text_files.txt").write_text("\n".join(skipped), encoding="utf-8")
    train_rows, val_rows, test_rows = [split_records(records, s) for s in ("train", "val", "test")]
    if not train_rows or not val_rows:
        raise RuntimeError("Need non-empty train and val records.")

    device = get_device(args.cpu)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    )
    max_len = resolve_max_len(model, args.max_len)
    model = to_device(model, device, not args.no_mgpu)

    print(f"Data root: {data_root.resolve()}")
    print(f"Labels ({len(labels)}): {', '.join(labels)}")
    print(f"Text samples: {len(records)} | Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)}")

    train_loader = DataLoader(TextDataset(train_rows, tokenizer, max_len), batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(TextDataset(val_rows, tokenizer, max_len), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(TextDataset(test_rows, tokenizer, max_len), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda") if test_rows else None

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, int(np.ceil(len(train_loader) / max(args.accum, 1))) * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(steps * args.warmup), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best, best_dir, history = -1.0, out / "best_auc_phobert", []
    for epoch in range(1, args.epochs + 1):
        train_loss = ce_epoch(model, train_loader, opt, sched, scaler, device, args.accum)
        val_metrics, val_ids, val_y, val_p, val_pred = eval_text(
            model,
            val_loader,
            device,
            args.threshold,
            label_names=labels,
            binary_positive_label=args.binary_positive_label,
        )
        val_score = val_metrics["auc"]
        if np.isnan(val_score):
            val_score = val_metrics["f1_macro"]
        history.append({
            "epoch": epoch,
            "train_loss": round_float(train_loss),
            "val_loss": round_float(val_metrics["loss"]),
            "val_acc": round_float(val_metrics["accuracy"]),
            "val_f1_macro": round_float(val_metrics["f1_macro"]),
            "val_f1_weighted": round_float(val_metrics["f1_weighted"]),
            "val_auc": round_float(val_metrics["auc"]),
            "val_score": round_float(val_score),
        })
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.3f} | "
            f"val_loss={val_metrics['loss']:.3f} | val_acc={val_metrics['accuracy']:.3f} | "
            f"val_f1_macro={val_metrics['f1_macro']:.3f} | val_auc={val_metrics['auc']:.3f}"
        )
        if val_score > best:
            best = val_score
            save_model(model, tokenizer, best_dir, epoch, val_metrics, args, max_len, labels, label_to_id)
            save_preds(
                out / "val_predictions_best_auc.csv",
                val_ids,
                val_y,
                val_p,
                val_pred,
                "id",
                label_names=labels,
                binary_positive_label=args.binary_positive_label,
            )

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
        metrics, ids, y, p, pred = eval_text(
            eval_model,
            loader,
            device,
            args.threshold,
            label_names=labels,
            binary_positive_label=args.binary_positive_label,
        )
        all_metrics[split] = round_metrics(metrics)
        save_preds(
            out / f"{split}_predictions_best_auc.csv",
            ids,
            y,
            p,
            pred,
            "id",
            label_names=labels,
            binary_positive_label=args.binary_positive_label,
        )
        print(format_metrics_summary(split, all_metrics[split]))
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved full metrics to {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
