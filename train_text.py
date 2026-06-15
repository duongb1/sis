import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sislib.common import quiet_hf_logging

quiet_hf_logging()

from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from sislib.common import get_device, resolve_max_len, round_float, round_metrics, seed_all, split_records, to_device, unwrap
from sislib.metrics import cls_metrics, format_metrics_summary, save_preds
from sislib.text_data import (
    EXCEL_TEXT_COLUMNS,
    FieldTextDataset,
    TextDataset,
    collect_excel_text,
    collect_large_text,
    collect_processed_csv_text,
    discover_excel_labels,
    discover_processed_csv_labels,
    discover_text_labels,
    is_excel_data,
    is_processed_csv_data,
    make_label_maps,
    parse_labels_arg,
    save_records,
)
from sislib.text_train import FieldAwarePhoBERTClassifier, PhoBERTClassifier, ce_epoch, eval_text


def parse_args():
    p = argparse.ArgumentParser(description="Train text-only PhoBERT on large text files.")
    p.add_argument("--data", default="/kaggle/input/datasets/duongb/cthsis/data/texts")
    p.add_argument("--out", default="/kaggle/working/text_phobert_classifier")
    p.add_argument("--model", default="vinai/phobert-base")
    p.add_argument("--format", choices=["auto", "text", "excel", "processed"], default="auto", help="Input format. processed accepts CSV files with Input_Text and Label columns.")
    p.add_argument("--excel-task", choices=["binary"], default="binary", help="For Excel input, train on co/khong binary targets.")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio for unsplit Excel input.")
    p.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio for unsplit Excel input.")
    p.add_argument("--split-strategy", choices=["random", "kfold"], default="random", help="Excel split strategy. kfold uses one fold as 20%% test when --n-folds 5.")
    p.add_argument("--n-folds", type=int, default=5, help="Number of folds for --split-strategy kfold.")
    p.add_argument("--fold-index", type=int, default=0, help="Zero-based held-out test fold for --split-strategy kfold.")
    p.add_argument("--excel-split-label", choices=["target", "binary"], default="binary", help="Label source used only to stratify Excel kfold splits.")
    p.add_argument("--labels", default=None, help="Comma-separated class names. Defaults to CSV/file labels discovered under --data.")
    p.add_argument("--binary-positive-label", default=None, help="Class treated as positive for one-vs-rest binary metrics. Defaults to co.")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--thresholds", default=None, help="Comma-separated thresholds for binary_positive_label metrics, e.g. 0.30,0.35,0.40,0.45,0.50.")

    p.add_argument("--pooling", choices=["cls", "attention", "gated"], default="cls", help="Pooling method after PhoBERT encoder. cls uses the first-token representation, attention learns token-level attention pooling, and gated fuses CLS with attention pooling.")
    p.add_argument("--input-mode", choices=["concat", "field"], default="concat", help="Input representation mode: concat all fields or encode Excel fields separately.")
    p.add_argument("--max-len-per-field", type=int, default=128, help="Maximum token length for each clinical field when --input-mode field.")
    p.add_argument("--save-field-attention", action="store_true", help="Save field-level attention weights in field-aware prediction CSVs.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    p.add_argument("--init-checkpoint", default=None, help="Optional checkpoint directory to initialize/fine-tune from.")
    p.add_argument("--eval-data", action="append", default=[], help="Extra evaluation data as NAME=PATH or PATH. Can be repeated.")
    p.add_argument("--eval-format", choices=["auto", "text", "excel", "processed"], default="auto")
    p.add_argument("--eval-split-strategy", choices=["eval", "random", "kfold"], default="eval")
    p.add_argument("--eval-splits", default="eval", help="Comma-separated splits to evaluate for --eval-data, e.g. eval or val,test.")
    return p.parse_args()


def parse_thresholds(value, default):
    if value is None:
        return [default]
    thresholds = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    return thresholds or [default]


def model_config_metadata(args):
    fusion = "cls_attention_gated" if args.pooling == "gated" else None
    config = {
        "pooling": args.pooling,
        "fusion": fusion,
        "primary_task": args.excel_task,
    }
    return {key: value for key, value in config.items() if value is not None}


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
                "pooling": args.pooling,
                "input_mode": args.input_mode,
                "max_len_per_field": args.max_len_per_field,
                "split_strategy": args.split_strategy,
                "n_folds": args.n_folds,
                "fold_index": args.fold_index,
                "excel_split_label": args.excel_split_label,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
                "model_config": model_config_metadata(args),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def save_state_dict_model(model, tokenizer, path, epoch, metrics, args, max_len, labels, label_to_id, best_model_metric="auc"):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(unwrap(model).state_dict(), path / "model.pt")
    tokenizer.save_pretrained(path)
    with open(path / "training_info.json", "w", encoding="utf-8") as f:
        info = {
                "epoch": epoch,
                "best_model_metric": best_model_metric,
                "val_metrics": round_metrics(metrics),
                "text_model_name": args.model,
                "max_length": max_len,
                "num_labels": len(labels),
                "labels": labels,
                "label_to_id": label_to_id,
                "id_to_label": {str(index): label for index, label in enumerate(labels)},
                "binary_positive_label": args.binary_positive_label,
                "pooling": args.pooling,
                "input_mode": args.input_mode,
                "max_len_per_field": args.max_len_per_field,
                "split_strategy": args.split_strategy,
                "n_folds": args.n_folds,
                "fold_index": args.fold_index,
                "excel_split_label": args.excel_split_label,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
                "model_config": model_config_metadata(args),
        }
        json.dump(info, f, ensure_ascii=False, indent=2)


def field_attention_rows(field_weights):
    if field_weights is None:
        return None
    rows = []
    for weights in field_weights:
        row = {}
        for field_name, weight in zip(EXCEL_TEXT_COLUMNS, weights):
            row[f"field_attention_{field_name}"] = round_float(weight, 6)
        row["top_field"] = EXCEL_TEXT_COLUMNS[int(np.argmax(weights))]
        rows.append(row)
    return rows


def label_distribution(rows, key):
    counts = Counter(row.get(key, "") for row in rows)
    counts.pop("", None)
    return dict(sorted(counts.items()))


def infer_input_format(data, requested):
    if requested != "auto":
        return requested
    if is_excel_data(data):
        return "excel"
    if is_processed_csv_data(data):
        return "processed"
    return "text"


def collect_records_for_args(args, labels, label_to_id, data=None, input_format=None, split_strategy=None, eval_split_name="eval"):
    data = args.data if data is None else data
    input_format = infer_input_format(data, args.format if input_format is None else input_format)
    split_strategy = args.split_strategy if split_strategy is None else split_strategy
    if input_format == "excel":
        return collect_excel_text(
            data,
            labels=labels,
            label_to_id=label_to_id,
            task=args.excel_task,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_strategy=split_strategy,
            n_folds=args.n_folds,
            fold_index=args.fold_index,
            split_label=args.excel_split_label,
        )
    if input_format == "processed":
        return collect_processed_csv_text(
            data,
            labels=labels,
            label_to_id=label_to_id,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_strategy=split_strategy,
            n_folds=args.n_folds,
            fold_index=args.fold_index,
            eval_split_name=eval_split_name,
        )
    if split_strategy not in {"random", "eval"}:
        raise RuntimeError(f"--split-strategy {split_strategy!r} is not supported for text folder input.")
    return collect_large_text(data, labels=labels, label_to_id=label_to_id)


def parse_eval_data_spec(spec):
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), path.strip()
    path = Path(spec)
    return path.stem, spec


def load_initial_checkpoint(model, checkpoint):
    checkpoint = Path(checkpoint)
    state_path = checkpoint / "model.pt"
    if state_path.exists():
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
        return True
    return False


def main():
    args = parse_args()
    seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    input_format = infer_input_format(args.data, args.format)
    use_excel = input_format == "excel"
    use_processed = input_format == "processed"
    if use_excel:
        labels = parse_labels_arg(args.labels) or discover_excel_labels(args.data, task="binary")
        if args.binary_positive_label is None:
            args.binary_positive_label = "co"
    elif use_processed:
        labels = parse_labels_arg(args.labels) or discover_processed_csv_labels(args.data)
        if args.binary_positive_label is None:
            args.binary_positive_label = "co"
    else:
        labels = parse_labels_arg(args.labels) or discover_text_labels(args.data)
        if args.binary_positive_label is None:
            args.binary_positive_label = "co"
    if len(labels) < 2:
        raise RuntimeError(f"Need at least two labels under --data, found: {labels}")
    label_to_id, id_to_label = make_label_maps(labels)

    records, skipped, data_root = collect_records_for_args(args, labels, label_to_id, input_format=input_format)
    if skipped:
        (out / "skipped_text_files.txt").write_text("\n".join(skipped), encoding="utf-8")
    train_rows, val_rows, test_rows = [split_records(records, s) for s in ("train", "val", "test")]
    if not train_rows or not val_rows:
        raise RuntimeError("Need non-empty train and val records.")
    if use_excel:
        print(f"Excel split stratify label: {args.excel_split_label}")
        for split_name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
            print(f"{split_name} label distribution: {label_distribution(rows, 'label_name')}")
    is_field_aware = args.input_mode == "field"
    if args.input_mode == "field":
        if not use_excel:
            raise RuntimeError("--input-mode field is currently supported only for Excel input.")
        if args.pooling == "gated":
            raise RuntimeError("--pooling gated is currently supported for --input-mode concat. Use --input-mode concat for gated fusion.")
    device = get_device(args.cpu)
    tokenizer_source = args.init_checkpoint if args.init_checkpoint and Path(args.init_checkpoint).exists() else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
    use_custom_classifier = args.pooling in {"attention", "gated"}
    hf_model_source = args.model
    if args.init_checkpoint and not (Path(args.init_checkpoint) / "model.pt").exists() and not (is_field_aware or use_custom_classifier):
        hf_model_source = args.init_checkpoint
    if is_field_aware:
        model = FieldAwarePhoBERTClassifier(
            args.model,
            num_labels=len(labels),
            num_fields=len(EXCEL_TEXT_COLUMNS),
            pooling=args.pooling,
        )
    elif use_custom_classifier:
        model = PhoBERTClassifier(
            args.model,
            num_labels=len(labels),
            pooling=args.pooling,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_source,
            num_labels=len(labels),
            id2label=id_to_label,
            label2id=label_to_id,
            ignore_mismatched_sizes=True,
        )
    max_len = resolve_max_len(model, args.max_len_per_field if is_field_aware else args.max_len)
    if args.init_checkpoint and Path(args.init_checkpoint).exists():
        loaded_state = load_initial_checkpoint(model, args.init_checkpoint)
        if loaded_state:
            print(f"Initialized model weights from {Path(args.init_checkpoint).resolve()}")
    model = to_device(model, device, not args.no_mgpu)

    print(f"Data root: {data_root.resolve()}")
    print(f"Labels ({len(labels)}): {', '.join(labels)}")
    print(f"Input mode: {args.input_mode}")
    print(f"Pooling: {args.pooling}")
    if args.pooling == "gated":
        print("Fusion: CLS + Attention gated fusion")
    print(f"Text samples: {len(records)} | Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)}")

    dataset_cls = FieldTextDataset if is_field_aware else TextDataset
    is_multi_gpu = device.type == "cuda" and not args.no_mgpu and torch.cuda.device_count() > 1
    train_loader = DataLoader(
        dataset_cls(train_rows, tokenizer, max_len),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=is_multi_gpu
    )
    val_loader = DataLoader(dataset_cls(val_rows, tokenizer, max_len), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(dataset_cls(test_rows, tokenizer, max_len), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda") if test_rows else None

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, int(np.ceil(len(train_loader) / max(args.accum, 1))) * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(steps * args.warmup), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best, best_dir, history = -1.0, out / "best_auc_phobert", []
    for epoch in range(1, args.epochs + 1):
        train_loss = ce_epoch(model, train_loader, opt, sched, scaler, device, args.accum)
        eval_result = eval_text(
            model,
            val_loader,
            device,
            args.threshold,
            label_names=labels,
            binary_positive_label=args.binary_positive_label,
            return_field_weights=is_field_aware and args.save_field_attention,
        )
        if is_field_aware and args.save_field_attention:
            val_metrics, val_ids, val_y, val_p, val_pred, val_field_weights = eval_result
        else:
            val_metrics, val_ids, val_y, val_p, val_pred = eval_result
            val_field_weights = None
        primary_val = val_metrics
        val_score = val_metrics["auc"]
        if np.isnan(val_score):
            val_score = val_metrics["f1_macro"]
        history.append({
            "epoch": epoch,
            "train_loss": round_float(train_loss),
            "val_loss": round_float(primary_val["loss"]),
            "val_acc": round_float(primary_val["accuracy"]),
            "val_f1_macro": round_float(primary_val["f1_macro"]),
            "val_f1_weighted": round_float(primary_val["f1_weighted"]),
            "val_auc": round_float(primary_val["auc"]),
            "val_score": round_float(val_score),
        })
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.3f} | "
            f"val_loss={primary_val['loss']:.3f} | val_acc={primary_val['accuracy']:.3f} | "
            f"val_f1_macro={primary_val['f1_macro']:.3f} | val_auc={primary_val['auc']:.3f}"
        )
        if val_score > best:
            best = val_score
            if use_custom_classifier or is_field_aware:
                save_state_dict_model(model, tokenizer, best_dir, epoch, val_metrics, args, max_len, labels, label_to_id, best_model_metric="auc")
                save_preds(
                    out / "val_predictions_best_auc.csv",
                    val_ids,
                    val_y,
                    val_p,
                    val_pred,
                    "id",
                    label_names=labels,
                    binary_positive_label=args.binary_positive_label,
                    threshold=args.threshold,
                    extra_rows=field_attention_rows(field_weights) if is_field_aware and args.save_field_attention else None,
                )
            else:
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
                    threshold=args.threshold,
                )

    with open(out / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    save_records(out / "dataset_records.csv", records)

    if is_field_aware:
        eval_model = FieldAwarePhoBERTClassifier(
            args.model,
            num_labels=len(labels),
            num_fields=len(EXCEL_TEXT_COLUMNS),
            pooling=args.pooling,
        )
        resolve_max_len(eval_model, max_len)
        eval_model.load_state_dict(torch.load(best_dir / "model.pt", map_location="cpu"))
        eval_model = to_device(eval_model, device, not args.no_mgpu)
    elif use_custom_classifier:
        eval_model = PhoBERTClassifier(
            args.model,
            num_labels=len(labels),
            pooling=args.pooling,
        )
        resolve_max_len(eval_model, max_len)
        eval_model.load_state_dict(torch.load(best_dir / "model.pt", map_location="cpu"))
        eval_model = to_device(eval_model, device, not args.no_mgpu)
    else:
        eval_model = to_device(AutoModelForSequenceClassification.from_pretrained(best_dir), device, not args.no_mgpu)
    all_metrics, threshold_sweeps = {}, {}
    eval_loaders = [("val", val_loader), ("test", test_loader)]
    extra_eval_records = {}
    for spec in args.eval_data:
        eval_name, eval_data = parse_eval_data_spec(spec)
        eval_format = infer_input_format(eval_data, args.eval_format)
        eval_records, eval_skipped, _ = collect_records_for_args(
            args,
            labels,
            label_to_id,
            data=eval_data,
            input_format=eval_format,
            split_strategy=args.eval_split_strategy,
            eval_split_name="eval",
        )
        if eval_skipped:
            (out / f"skipped_{eval_name}_files.txt").write_text("\n".join(eval_skipped), encoding="utf-8")
        wanted_splits = [item.strip() for item in args.eval_splits.split(",") if item.strip()]
        for split_name in wanted_splits:
            rows = split_records(eval_records, split_name)
            if not rows:
                continue
            extra_eval_records[f"{eval_name}_{split_name}"] = rows
            eval_loaders.append(
                (
                    f"{eval_name}_{split_name}",
                    DataLoader(dataset_cls(rows, tokenizer, max_len), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda"),
                )
            )
    if extra_eval_records:
        print("Extra evaluation splits:")
        for split_name, rows in extra_eval_records.items():
            print(f"{split_name}: {len(rows)}")
    for split, loader in eval_loaders:
        if loader is None:
            continue
        eval_result = eval_text(
            eval_model,
            loader,
            device,
            args.threshold,
            label_names=labels,
            binary_positive_label=args.binary_positive_label,
            return_field_weights=is_field_aware and args.save_field_attention,
        )
        if is_field_aware and args.save_field_attention:
            metrics, ids, y, p, pred, field_weights = eval_result
        else:
            metrics, ids, y, p, pred = eval_result
            field_weights = None
        all_metrics[split] = round_metrics(metrics)
        threshold_sweeps[split] = {}
        for threshold in parse_thresholds(args.thresholds, args.threshold):
            sweep_pred = (p[:, 1] >= threshold).astype(np.int64) if p.ndim == 2 and p.shape[1] == 2 else pred
            sweep_metrics = cls_metrics(
                y,
                p,
                sweep_pred,
                threshold=threshold,
                label_names=labels,
                binary_positive_label=args.binary_positive_label,
            )
            threshold_sweeps[split][str(threshold)] = round_metrics(sweep_metrics.get("binary_i63", sweep_metrics))
        save_preds(
            out / f"{split}_predictions_best_auc.csv",
            ids,
            y,
            p,
            pred,
            "id",
            label_names=labels,
            binary_positive_label=args.binary_positive_label,
            threshold=args.threshold,
            extra_rows=field_attention_rows(field_weights) if is_field_aware and args.save_field_attention else None,
        )
        print(format_metrics_summary(split, all_metrics[split]))
    metrics_payload = {
        **all_metrics,
        "binary_threshold_sweep": threshold_sweeps,
        "model_config": model_config_metadata(args),
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
    print(f"Saved full metrics to {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
