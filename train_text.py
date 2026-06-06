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
    EXCEL_MULTICLASS_LABELS,
    EXCEL_TEXT_COLUMNS,
    FieldTextDataset,
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
from sislib.text_train import FieldAwarePhoBERTClassifier, HardNegativeSupConLoss, PhoBERTClassifier, PhoBERTMultiTask, ce_epoch, eval_multitask, eval_text, multitask_epoch


def parse_args():
    p = argparse.ArgumentParser(description="Train text-only PhoBERT on large text files.")
    p.add_argument("--data", default="/kaggle/input/datasets/duongb/cthsis/data/texts")
    p.add_argument("--out", default="/kaggle/working/text_phobert_classifier")
    p.add_argument("--model", default="vinai/phobert-base")
    p.add_argument("--format", choices=["auto", "text", "excel"], default="auto", help="Input format. Excel accepts one .xlsx file, comma-separated .xlsx files, or a folder of .xlsx files.")
    p.add_argument("--excel-task", choices=["multiclass", "binary", "multitask"], default="multiclass", help="For Excel input, train on LABEL multi-class targets, co/khong binary targets, or binary + 3-class auxiliary multi-task targets.")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio for unsplit Excel input.")
    p.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio for unsplit Excel input.")
    p.add_argument("--split-strategy", choices=["random", "kfold"], default="random", help="Excel split strategy. kfold uses one fold as 20%% test when --n-folds 5.")
    p.add_argument("--n-folds", type=int, default=5, help="Number of folds for --split-strategy kfold.")
    p.add_argument("--fold-index", type=int, default=0, help="Zero-based held-out test fold for --split-strategy kfold.")
    p.add_argument("--excel-split-label", choices=["target", "binary", "multiclass"], default="multiclass", help="Label source used only to stratify Excel kfold splits.")
    p.add_argument("--labels", default=None, help="Comma-separated class names. Defaults to CSV/file labels discovered under --data.")
    p.add_argument("--binary-positive-label", default=None, help="Class treated as positive for one-vs-rest binary metrics. Defaults to I63_INFARCTION, or co for --excel-task binary.")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--thresholds", default=None, help="Comma-separated thresholds for binary_positive_label metrics, e.g. 0.30,0.35,0.40,0.45,0.50.")
    p.add_argument("--lambda-aux", "--aux-weight", dest="lambda_aux", type=float, default=0.5, help="Auxiliary 3-class loss weight for --excel-task multitask.")
    p.add_argument("--contrastive-loss", choices=["none", "hard_supcon"], default="none", help="Optional auxiliary contrastive loss for binary Excel training.")
    p.add_argument("--contrastive-weight", type=float, default=0.0)
    p.add_argument("--contrastive-temperature", type=float, default=0.1)
    p.add_argument("--hard-negative-weight", type=float, default=2.0)
    p.add_argument("--contrastive-proj-dim", type=int, default=128)
    p.add_argument("--pooling", choices=["cls", "attention", "gated"], default="cls", help="Pooling method after PhoBERT encoder. cls uses the first-token representation, attention learns token-level attention pooling, and gated fuses CLS with attention pooling.")
    p.add_argument("--input-mode", choices=["concat", "field"], default="concat", help="Input representation mode: concat all fields or encode Excel fields separately.")
    p.add_argument("--max-len-per-field", type=int, default=128, help="Maximum token length for each clinical field when --input-mode field.")
    p.add_argument("--save-field-attention", action="store_true", help="Save field-level attention weights in field-aware prediction CSVs.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
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
        "aux_weight": args.lambda_aux if args.excel_task == "multitask" else None,
        "primary_task": "binary_i63" if args.excel_task == "multitask" else args.excel_task,
        "aux_task": "3class_disease_structure" if args.excel_task == "multitask" else None,
    }
    return {key: value for key, value in config.items() if value is not None}


def contrastive_metadata(args):
    return {
        "loss": args.contrastive_loss,
        "weight": args.contrastive_weight,
        "temperature": args.contrastive_temperature,
        "hard_negative_weight": args.hard_negative_weight,
        "projection_dim": args.contrastive_proj_dim,
        "hard_negative_pair": ["I63_INFARCTION", "OTHER_STROKE_LIKE"],
    }


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
                "contrastive": contrastive_metadata(args),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def save_state_dict_model(model, tokenizer, path, epoch, metrics, args, max_len, labels, label_to_id, aux_labels=None, best_model_metric="auc"):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(unwrap(model).state_dict(), path / "model.pt")
    tokenizer.save_pretrained(path)
    aux_labels = list(aux_labels or [])
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
                "contrastive": contrastive_metadata(args),
        }
        if aux_labels:
            info.update(
                {
                    "aux_labels": aux_labels,
                    "aux_label_to_id": {label: index for index, label in enumerate(aux_labels)},
                    "aux_id_to_label": {str(index): label for index, label in enumerate(aux_labels)},
                    "lambda_aux": args.lambda_aux,
                }
            )
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
            split_label=args.excel_split_label,
        )
    else:
        records, skipped, data_root = collect_large_text(args.data, labels=labels, label_to_id=label_to_id)
    if skipped:
        (out / "skipped_text_files.txt").write_text("\n".join(skipped), encoding="utf-8")
    train_rows, val_rows, test_rows = [split_records(records, s) for s in ("train", "val", "test")]
    if not train_rows or not val_rows:
        raise RuntimeError("Need non-empty train and val records.")
    if use_excel:
        print(f"Excel split stratify label: {args.excel_split_label}")
        for split_name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
            print(f"{split_name} multiclass distribution: {label_distribution(rows, 'multiclass_label_name')}")
    is_multitask = use_excel and args.excel_task == "multitask"
    is_field_aware = args.input_mode == "field"
    if args.input_mode == "field":
        if not use_excel:
            raise RuntimeError("--input-mode field is currently supported only for Excel input.")
        if args.excel_task == "multitask":
            raise RuntimeError("--input-mode field is currently supported for binary/multiclass Excel tasks, not multitask.")
        if args.pooling == "gated":
            raise RuntimeError("--pooling gated is currently supported for --input-mode concat. Use --input-mode concat for gated fusion.")
    contrastive_enabled = args.contrastive_loss == "hard_supcon" and args.contrastive_weight > 0
    if contrastive_enabled:
        if not use_excel:
            raise RuntimeError("--contrastive-loss hard_supcon requires Excel input with multiclass LABEL.")
        if args.excel_task != "binary":
            raise RuntimeError("--contrastive-loss hard_supcon is currently supported for --excel-task binary only.")
        if is_field_aware:
            raise RuntimeError("--contrastive-loss hard_supcon requires --input-mode concat.")

    device = get_device(args.cpu)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    use_custom_classifier = (args.pooling in {"attention", "gated"} or contrastive_enabled) and not is_multitask
    aux_labels = list(EXCEL_MULTICLASS_LABELS)
    if is_field_aware:
        model = FieldAwarePhoBERTClassifier(
            args.model,
            num_labels=len(labels),
            num_fields=len(EXCEL_TEXT_COLUMNS),
            pooling=args.pooling,
        )
    elif is_multitask:
        model = PhoBERTMultiTask(args.model, pooling=args.pooling)
    elif use_custom_classifier:
        model = PhoBERTClassifier(
            args.model,
            num_labels=len(labels),
            pooling=args.pooling,
            contrastive_proj_dim=args.contrastive_proj_dim if contrastive_enabled else None,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=len(labels),
            id2label=id_to_label,
            label2id=label_to_id,
            ignore_mismatched_sizes=True,
        )
    max_len = resolve_max_len(model, args.max_len_per_field if is_field_aware else args.max_len)
    model = to_device(model, device, not args.no_mgpu)

    print(f"Data root: {data_root.resolve()}")
    print(f"Labels ({len(labels)}): {', '.join(labels)}")
    if is_multitask:
        print(f"Primary labels ({len(labels)}): {', '.join(labels)}")
        print(f"Aux labels ({len(aux_labels)}): {', '.join(aux_labels)}")
        print(f"Multi-task loss: loss = loss_binary + {args.lambda_aux:g} * loss_aux")
    print(f"Input mode: {args.input_mode}")
    print(f"Pooling: {args.pooling}")
    if args.pooling == "gated":
        print("Fusion: CLS + Attention gated fusion")
    if contrastive_enabled:
        print("Contrastive loss: hard_supcon")
        print(f"Contrastive weight: {args.contrastive_weight:g}")
        print(f"Contrastive temperature: {args.contrastive_temperature:g}")
        print(f"Hard negative weight: {args.hard_negative_weight:g}")
        print(f"Contrastive projection dim: {args.contrastive_proj_dim}")
    print(f"Text samples: {len(records)} | Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)}")

    dataset_cls = FieldTextDataset if is_field_aware else TextDataset
    train_loader = DataLoader(dataset_cls(train_rows, tokenizer, max_len), batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(dataset_cls(val_rows, tokenizer, max_len), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(dataset_cls(test_rows, tokenizer, max_len), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda") if test_rows else None

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, int(np.ceil(len(train_loader) / max(args.accum, 1))) * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(steps * args.warmup), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    contrastive_loss_fn = None
    if contrastive_enabled:
        aux_label_to_id = {label: index for index, label in enumerate(EXCEL_MULTICLASS_LABELS)}
        contrastive_loss_fn = HardNegativeSupConLoss(
            temperature=args.contrastive_temperature,
            hard_negative_weight=args.hard_negative_weight,
            i63_label_id=aux_label_to_id["I63_INFARCTION"],
            other_stroke_like_label_id=aux_label_to_id["OTHER_STROKE_LIKE"],
        )

    best, best_dir, history = -1.0, out / "best_auc_phobert", []
    for epoch in range(1, args.epochs + 1):
        if is_multitask:
            train_loss = multitask_epoch(model, train_loader, opt, sched, scaler, device, args.accum, args.lambda_aux)
            val_metrics, val_ids, val_y, val_p, val_pred, val_aux_y, val_aux_p, val_aux_pred = eval_multitask(
                model,
                val_loader,
                device,
                args.threshold,
                binary_label_names=labels,
                aux_label_names=aux_labels,
                lambda_aux=args.lambda_aux,
            )
            primary_val = val_metrics["primary_binary"]
            val_score = primary_val["auc"]
            if np.isnan(val_score):
                val_score = primary_val["f1_macro"]
        else:
            train_stats = ce_epoch(
                model,
                train_loader,
                opt,
                sched,
                scaler,
                device,
                args.accum,
                contrastive_loss_fn=contrastive_loss_fn,
                contrastive_weight=args.contrastive_weight if contrastive_enabled else 0.0,
            )
            if isinstance(train_stats, dict):
                train_loss = train_stats["total"]
                train_binary_loss = train_stats["binary"]
                train_contrastive_loss = train_stats["contrastive"]
            else:
                train_loss = train_stats
                train_binary_loss = train_loss
                train_contrastive_loss = 0.0
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
        if not is_multitask:
            history[-1]["train_binary_loss"] = round_float(train_binary_loss)
            history[-1]["train_contrastive_loss"] = round_float(train_contrastive_loss)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.3f} | "
            f"val_loss={primary_val['loss']:.3f} | val_acc={primary_val['accuracy']:.3f} | "
            f"val_f1_macro={primary_val['f1_macro']:.3f} | val_auc={primary_val['auc']:.3f}"
        )
        if contrastive_enabled:
            print(
                f"  train_binary_loss={train_binary_loss:.3f} | "
                f"train_contrastive_loss={train_contrastive_loss:.3f} | "
                f"train_total_loss={train_loss:.3f}"
            )
        if val_score > best:
            best = val_score
            if is_multitask:
                save_state_dict_model(model, tokenizer, best_dir, epoch, val_metrics, args, max_len, labels, label_to_id, aux_labels, best_model_metric="primary_binary_auc")
                save_preds(out / "val_predictions_best_auc.csv", val_ids, val_y, val_p, val_pred, "id", label_names=labels, binary_positive_label=args.binary_positive_label, threshold=args.threshold)
                save_preds(out / "val_aux_predictions_best_auc.csv", val_ids, val_aux_y, val_aux_p, val_aux_pred, "id", label_names=aux_labels, binary_positive_label=args.binary_positive_label, threshold=args.threshold)
            elif use_custom_classifier or is_field_aware:
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
                    extra_rows=field_attention_rows(val_field_weights) if is_field_aware and args.save_field_attention else None,
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

    if is_multitask:
        eval_model = PhoBERTMultiTask(args.model, pooling=args.pooling)
        resolve_max_len(eval_model, max_len)
        eval_model.load_state_dict(torch.load(best_dir / "model.pt", map_location="cpu"))
        eval_model = to_device(eval_model, device, not args.no_mgpu)
    elif is_field_aware:
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
            contrastive_proj_dim=args.contrastive_proj_dim if contrastive_enabled else None,
        )
        resolve_max_len(eval_model, max_len)
        eval_model.load_state_dict(torch.load(best_dir / "model.pt", map_location="cpu"))
        eval_model = to_device(eval_model, device, not args.no_mgpu)
    else:
        eval_model = to_device(AutoModelForSequenceClassification.from_pretrained(best_dir), device, not args.no_mgpu)
    all_metrics, threshold_sweeps = {}, {}
    for split, loader in [("val", val_loader), ("test", test_loader)]:
        if loader is None:
            continue
        if is_multitask:
            metrics, ids, y, p, pred, aux_y, aux_p, aux_pred = eval_multitask(
                eval_model,
                loader,
                device,
                args.threshold,
                binary_label_names=labels,
                aux_label_names=aux_labels,
                lambda_aux=args.lambda_aux,
            )
            all_metrics[split] = round_metrics(metrics)
            threshold_sweeps[split] = {}
            for threshold in parse_thresholds(args.thresholds, args.threshold):
                sweep_pred = (p[:, 1] >= threshold).astype(np.int64)
                sweep_metrics = cls_metrics(
                    y,
                    p,
                    sweep_pred,
                    threshold=threshold,
                    label_names=labels,
                    binary_positive_label=args.binary_positive_label,
                )
                threshold_sweeps[split][str(threshold)] = round_metrics(sweep_metrics["binary_i63"])
            save_preds(out / f"{split}_predictions_best_auc.csv", ids, y, p, pred, "id", label_names=labels, binary_positive_label=args.binary_positive_label, threshold=args.threshold)
            save_preds(out / f"{split}_aux_predictions_best_auc.csv", ids, aux_y, aux_p, aux_pred, "id", label_names=aux_labels, binary_positive_label=args.binary_positive_label, threshold=args.threshold)
            print(format_metrics_summary(f"{split}.primary_binary", all_metrics[split]["primary_binary"]))
            print(format_metrics_summary(f"{split}.aux_3class", all_metrics[split]["aux_3class"]))
        else:
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
                sweep_metrics = cls_metrics(
                    y,
                    p,
                    pred,
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
        "contrastive": contrastive_metadata(args),
    }
    if is_multitask:
        metrics_payload["selection"] = {
            "checkpoint_metric": "primary_binary_auc",
            "uses_auxiliary_metric_for_selection": False,
            "inference_head": "primary_binary",
            "lambda_aux": args.lambda_aux,
        }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
    print(f"Saved full metrics to {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
