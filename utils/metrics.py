import csv
import json
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from .common import ID_TO_LABEL, round_float


def _safe_auc(labels, probs, num_classes):
    try:
        if num_classes == 2:
            scores = probs[:, 1] if probs.ndim == 2 else probs
            return float(roc_auc_score(labels, scores))
        return float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def _expected_calibration_error(labels, probs, num_bins=10):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    if len(labels) == 0:
        return 0.0
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == num_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
    return float(ece)


def _binary_one_vs_rest_metrics(labels, probs, positive_index, positive_label, threshold=0.5):
    true_binary = (labels == positive_index).astype(np.int64)
    if probs.ndim == 2:
        positive_probs = probs[:, positive_index]
    else:
        positive_probs = probs
    pred_binary = (positive_probs >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(true_binary, pred_binary, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    brier = float(np.mean((positive_probs - true_binary) ** 2))
    ece = _expected_calibration_error(true_binary, positive_probs)
    return {
        "positive_label": positive_label,
        "negative_label": f"NOT_{positive_label}",
        "decision_rule": f"P({positive_label}) >= threshold",
        "threshold": threshold,
        "accuracy": float(accuracy_score(true_binary, pred_binary)),
        "f1": float(f1_score(true_binary, pred_binary, zero_division=0)),
        "auc": _safe_auc(true_binary, positive_probs, 2),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier_score": brier,
        "ece": ece,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "num_positive": int(true_binary.sum()),
        "num_negative": int((true_binary == 0).sum()),
    }


def cls_metrics(
    labels,
    probs,
    preds,
    loss=None,
    id_name="num_samples",
    threshold=0.5,
    label_names=None,
    binary_positive_label=None,
):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)
    preds = np.asarray(preds, dtype=np.int64)
    if probs.ndim == 1:
        num_classes = 2
    else:
        num_classes = probs.shape[1]
    label_ids = list(range(num_classes))
    label_names = list(label_names) if label_names is not None else [ID_TO_LABEL.get(i, str(i)) for i in label_ids]
    cm = confusion_matrix(labels, preds, labels=label_ids).tolist()
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "auc": _safe_auc(labels, probs, num_classes),
        "confusion_matrix": cm,
        id_name: int(len(labels)),
        "num_classes": int(num_classes),
        "class_names": label_names,
        "class_counts": {label_names[i]: int((labels == i).sum()) for i in label_ids},
        "threshold": threshold,
    }
    if num_classes == 2:
        tn, fp = cm[0]
        fn, tp = cm[1]
        sens = float(tp / (tp + fn)) if (tp + fn) else float("nan")
        spec = float(tn / (tn + fp)) if (tn + fp) else float("nan")
        if probs.ndim == 2:
            pos_probs = probs[:, 1]
        else:
            pos_probs = probs
        brier = float(np.mean((pos_probs - labels) ** 2))
        ece_val = _expected_calibration_error(labels, pos_probs)
        metrics.update(
            {
                "sensitivity": sens,
                "specificity": spec,
                "brier_score": brier,
                "ece": ece_val,
            }
        )
    if binary_positive_label and binary_positive_label in label_names:
        positive_index = label_names.index(binary_positive_label)
        metrics["binary_i63"] = _binary_one_vs_rest_metrics(labels, probs, positive_index, binary_positive_label, threshold)
    if loss is not None:
        metrics = {"loss": float(loss), **metrics}
    return metrics


def save_preds(path, ids, labels, probs, preds, id_field, label_names=None, binary_positive_label=None, threshold=0.5, extra_rows=None):
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim == 1:
        probs = np.stack([1.0 - probs, probs], axis=1)
    num_classes = probs.shape[1]
    label_names = list(label_names) if label_names is not None else [ID_TO_LABEL.get(i, str(i)) for i in range(num_classes)]
    prob_fields = [f"prob_{label}" for label in label_names]
    binary_fields = []
    positive_index = None
    if binary_positive_label and binary_positive_label in label_names:
        positive_index = label_names.index(binary_positive_label)
        binary_fields = ["true_binary_i63", "pred_binary_i63", "prob_binary_i63"]
    extra_rows = list(extra_rows or [])
    extra_fields = sorted(set().union(*(row.keys() for row in extra_rows))) if extra_rows else []

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[id_field, "true_label", "true_name", "pred_label", "pred_name", *binary_fields, *prob_fields, *extra_fields],
        )
        writer.writeheader()
        for index, (item_id, label, prob_row, pred) in enumerate(zip(ids, labels, probs, preds)):
            row = {
                id_field: item_id,
                "true_label": int(label),
                "true_name": label_names[int(label)] if int(label) < len(label_names) else str(label),
                "pred_label": int(pred),
                "pred_name": label_names[int(pred)] if int(pred) < len(label_names) else str(pred),
            }
            for field, prob in zip(prob_fields, prob_row):
                row[field] = round_float(prob)
            if positive_index is not None:
                row["true_binary_i63"] = int(int(label) == positive_index)
                row["pred_binary_i63"] = int(prob_row[positive_index] >= threshold)
                row["prob_binary_i63"] = round_float(prob_row[positive_index])
            if extra_rows:
                row.update(extra_rows[index])
            writer.writerow(row)


def format_metrics_summary(name, metrics):
    parts = [
        f"{name}:",
        f"loss={metrics.get('loss', float('nan'))}",
        f"acc={metrics.get('accuracy', float('nan'))}",
        f"f1_macro={metrics.get('f1_macro', metrics.get('f1', float('nan')))}",
        f"f1_weighted={metrics.get('f1_weighted', float('nan'))}",
        f"auc={metrics.get('auc', float('nan'))}",
    ]
    return " ".join(str(part) for part in parts)


# --- REPORTS & AGGREGATION HELPERS ---

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_numeric(prefix, value, out):
    if isinstance(value, dict):
        for key, child in value.items():
            flatten_numeric(f"{prefix}.{key}" if prefix else key, child, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)


def summarize_metric_rows(rows):
    keys = sorted(set().union(*(row.keys() for row in rows)))
    summary = {}
    for key in keys:
        values = np.array([row[key] for row in rows if key in row and not np.isnan(row[key])], dtype=np.float64)
        if values.size == 0:
            continue
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "n": int(values.size),
        }
    return summary


def write_summary_csv(path, summary):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "n", "mean_plus_minus_std"])
        writer.writeheader()
        for metric, stats in summary.items():
            writer.writerow(
                {
                    "metric": metric,
                    "mean": f"{stats['mean']:.6f}",
                    "std": f"{stats['std']:.6f}",
                    "n": stats["n"],
                    "mean_plus_minus_std": f"{stats['mean']:.3f} ± {stats['std']:.3f}",
                }
            )


def print_key_summary(name, summary):
    keys = [
        "test.accuracy",
        "test.f1",
        "test.f1_macro",
        "test.f1_weighted",
        "test.auc",
        "test.sensitivity",
        "test.specificity",
        "test.brier_score",
        "test.ece",
    ]
    print(f"5-fold mean ± std: {name}")
    for key in keys:
        stats = summary.get(key)
        if stats:
            print(f"{key}: {stats['mean']:.3f} ± {stats['std']:.3f}")


def extract_test_binary_metrics(metrics):
    test = metrics.get("test", {})
    if "primary_binary" in test:
        primary = test["primary_binary"]
        if "binary_i63" in primary:
            return dict(primary["binary_i63"])
        return dict(primary)
    if "binary_i63" in test:
        return dict(test["binary_i63"])
    return dict(test)


def confusion_counts(metrics):
    if "confusion_matrix" in metrics:
        (tn, fp), (fn, tp) = metrics["confusion_matrix"]
        return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return {key: int(metrics.get(key, 0)) for key in ("tn", "fp", "fn", "tp")}


def collect_model_report(output_dir, folder, folds):
    rows = []
    counts = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for fold in range(folds):
        path = output_dir / folder / f"fold_{fold}" / "metrics.json"
        if not path.exists():
            return None
        data = load_json(path)
        metrics = extract_test_binary_metrics(data)
        row = {f"test.{key}": value for key, value in metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
        selected = data.get("selected", {})
        for key in ("threshold",):
            if key in selected:
                row[f"selected.{key}"] = float(selected[key])
        if "val" in data:
            val_metrics = dict(data["val"])
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row[f"selected.val.{key}"] = float(value)
        rows.append(row)
        fold_counts = confusion_counts(metrics)
        for key in counts:
            counts[key] += fold_counts[key]
    return {"summary": summarize_metric_rows(rows), "counts": counts}


def mean_std_text(summary, key):
    stats = summary.get(key)
    if not stats:
        return "n/a"
    return f"{stats['mean']:.3f}±{stats['std']:.3f}"


def best_model_line(reports, metric_key):
    candidates = [(name, data["summary"].get(metric_key, {}).get("mean")) for name, data in reports.items()]
    candidates = [(name, value) for name, value in candidates if value is not None and not np.isnan(value)]
    if not candidates:
        return None
    best_value = max(value for _, value in candidates)
    winners = [name for name, value in candidates if abs(value - best_value) < 1e-12]
    return ", ".join(winners)


def print_final_small_report(output_dir, args):
    model_folders = [
        ("small_binary", "small_binary"),
    ]
    reports = {}
    for display_name, folder in model_folders:
        report = collect_model_report(output_dir, folder, args.folds)
        if report:
            reports[display_name] = report
    if not reports:
        return

    print("\n" + "-" * 80)
    print("5-fold summary: small models")
    print("Model                         Acc       F1        AUC       Sens      Spec      Brier     ECE")
    for name, report in reports.items():
        summary = report["summary"]
        print(
            f"{name:<29} "
            f"{mean_std_text(summary, 'test.accuracy'):<9} "
            f"{mean_std_text(summary, 'test.f1'):<9} "
            f"{mean_std_text(summary, 'test.auc'):<9} "
            f"{mean_std_text(summary, 'test.sensitivity'):<9} "
            f"{mean_std_text(summary, 'test.specificity'):<9} "
            f"{mean_std_text(summary, 'test.brier_score'):<9} "
            f"{mean_std_text(summary, 'test.ece'):<9}"
        )

    print("\nAggregate confusion counts:")
    for name, report in reports.items():
        counts = report["counts"]
        print(f"{name:<29} TN={counts['tn']} FP={counts['fp']} FN={counts['fn']} TP={counts['tp']}")


def aggregate_experiment(output_dir, experiment_name, folds):
    rows = []
    missing = []
    for fold in range(folds):
        metrics_path = output_dir / experiment_name / f"fold_{fold}" / "metrics.json"
        if not metrics_path.exists():
            missing.append(str(metrics_path))
            continue
        flattened = {}
        flatten_numeric("", load_json(metrics_path), flattened)
        rows.append(flattened)

    if missing:
        print(f"Skip summary for {experiment_name}: missing {len(missing)} metrics files.")
        return None
    if not rows:
        return None

    summary = summarize_metric_rows(rows)
    summary_dir = output_dir / experiment_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / "summary_5fold.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_summary_csv(summary_dir / "summary_5fold.csv", summary)
    print_key_summary(experiment_name, summary)
    return summary
