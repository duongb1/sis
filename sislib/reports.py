import csv
import json

import numpy as np


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
        "test.balanced_accuracy",
        "test.binary_i63.accuracy",
        "test.binary_i63.f1",
        "test.binary_i63.auc",
        "test.binary_i63.sensitivity",
        "test.binary_i63.specificity",
        "test.binary_i63.balanced_accuracy",
        "test.primary_binary.accuracy",
        "test.primary_binary.f1",
        "test.primary_binary.auc",
        "test.primary_binary.sensitivity",
        "test.primary_binary.specificity",
        "test.primary_binary.balanced_accuracy",
        "test.aux_3class.accuracy",
        "test.aux_3class.f1_macro",
        "test.aux_3class.auc",
    ]
    print("\n" + "-" * 80)
    print(f"5-fold mean ± std: {name}")
    for key in keys:
        stats = summary.get(key)
        if stats:
            print(f"{key}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    for key in sorted(summary):
        if key.startswith("binary_threshold_sweep.test."):
            stats = summary[key]
            print(f"{key}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    for key in [
        "selected.threshold",
        "selected.val.sensitivity",
        "selected.val.specificity",
        "selected.val.balanced_accuracy",
        "selected.metrics.sensitivity",
        "selected.metrics.specificity",
    ]:
        stats = summary.get(key)
        if stats:
            print(f"{key}: {stats['mean']:.3f} ± {stats['std']:.3f}")


def with_balanced_accuracy(metrics):
    metrics = dict(metrics)
    if "balanced_accuracy" not in metrics and "sensitivity" in metrics and "specificity" in metrics:
        metrics["balanced_accuracy"] = float((metrics["sensitivity"] + metrics["specificity"]) / 2.0)
    return metrics


def extract_test_binary_metrics(metrics):
    test = metrics.get("test", {})
    if "primary_binary" in test:
        primary = test["primary_binary"]
        if "binary_i63" in primary:
            return with_balanced_accuracy(primary["binary_i63"])
        return with_balanced_accuracy(primary)
    if "binary_i63" in test:
        return with_balanced_accuracy(test["binary_i63"])
    return with_balanced_accuracy(test)


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
            val_metrics = with_balanced_accuracy(data["val"])
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


def print_threshold_sweep_table(output_dir, folder, folds):
    rows_by_threshold = {}
    for fold in range(folds):
        path = output_dir / folder / f"fold_{fold}" / "metrics.json"
        if not path.exists():
            return
        sweep = load_json(path).get("binary_threshold_sweep", {}).get("test", {})
        for threshold, metrics in sweep.items():
            metrics = with_balanced_accuracy(metrics)
            rows_by_threshold.setdefault(threshold, []).append(
                {f"test.{key}": value for key, value in metrics.items() if key in {"accuracy", "f1", "auc", "sensitivity", "specificity", "balanced_accuracy"}}
            )
    if not rows_by_threshold:
        return
    print(f"\nThreshold sweep: {folder}")
    print("threshold | acc       | f1        | auc       | sens      | spec      | bal_acc")
    for threshold in sorted(rows_by_threshold, key=lambda item: float(item)):
        summary = summarize_metric_rows(rows_by_threshold[threshold])
        print(
            f"{float(threshold):>9.2f} | "
            f"{mean_std_text(summary, 'test.accuracy'):<9} | "
            f"{mean_std_text(summary, 'test.f1'):<9} | "
            f"{mean_std_text(summary, 'test.auc'):<9} | "
            f"{mean_std_text(summary, 'test.sensitivity'):<9} | "
            f"{mean_std_text(summary, 'test.specificity'):<9} | "
            f"{mean_std_text(summary, 'test.balanced_accuracy'):<9}"
        )


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
        ("small_multiclass_to_binary", "small_multiclass"),
        ("small_multitask", "small_multitask"),
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
    print("Model                         Acc       F1        AUC       Sens      Spec      BalAcc")
    for name, report in reports.items():
        summary = report["summary"]
        print(
            f"{name:<29} "
            f"{mean_std_text(summary, 'test.accuracy'):<9} "
            f"{mean_std_text(summary, 'test.f1'):<9} "
            f"{mean_std_text(summary, 'test.auc'):<9} "
            f"{mean_std_text(summary, 'test.sensitivity'):<9} "
            f"{mean_std_text(summary, 'test.specificity'):<9} "
            f"{mean_std_text(summary, 'test.balanced_accuracy'):<9}"
        )

    print("\nAggregate confusion counts:")
    for name, report in reports.items():
        counts = report["counts"]
        print(f"{name:<29} TN={counts['tn']} FP={counts['fp']} FN={counts['fn']} TP={counts['tp']}")

    baseline = reports.get("small_binary", {}).get("counts")
    if baseline:
        print("\nFP/FN trade-off vs small_binary:")
        for name, report in reports.items():
            if name == "small_binary":
                continue
            counts = report["counts"]
            print(f"{name:<29} FP {counts['fp'] - baseline['fp']:+d}, FN {counts['fn'] - baseline['fn']:+d}")

    print("\nBest metrics:")
    for label, key in [
        ("best_f1", "test.f1"),
        ("best_auc", "test.auc"),
        ("best_sensitivity", "test.sensitivity"),
        ("best_specificity", "test.specificity"),
        ("best_balanced_accuracy", "test.balanced_accuracy"),
    ]:
        winner = best_model_line(reports, key)
        if winner:
            print(f"{label}: {winner}")

    print_threshold_sweep_table(output_dir, "small_binary", args.folds)
    print_threshold_sweep_table(output_dir, "small_multiclass", args.folds)
    print_threshold_sweep_table(output_dir, "small_multitask", args.folds)


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
