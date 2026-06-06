import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_EXCEL_ROOT = "/kaggle/input/datasets/duongbui/siscth"


EXPERIMENTS = [
    {
        "name": "large_binary",
        "data": "9937_co_label.xlsx,9937_khong_label.xlsx",
        "task": "binary",
        "positive": "co",
    },
    {
        "name": "small_binary",
        "data": "700_co_label.xlsx,700_khong_label.xlsx",
        "task": "binary",
        "positive": "co",
    },
    {
        "name": "large_multiclass",
        "data": "9937_co_label.xlsx,9937_khong_label.xlsx",
        "task": "multiclass",
        "positive": "I63_INFARCTION",
    },
    {
        "name": "large_multitask",
        "data": "9937_co_label.xlsx,9937_khong_label.xlsx",
        "task": "multitask",
        "positive": "I63_INFARCTION",
    },
    {
        "name": "small_multiclass",
        "data": "700_co_label.xlsx,700_khong_label.xlsx",
        "task": "multiclass",
        "positive": "I63_INFARCTION",
    },
    {
        "name": "small_multitask",
        "data": "700_co_label.xlsx,700_khong_label.xlsx",
        "task": "multitask",
        "positive": "I63_INFARCTION",
    },
]

EXPERIMENT_ALIASES = {
    "large": {"large_binary", "large_multiclass", "large_multitask"},
    "small": {"small_binary", "small_multiclass", "small_multitask"},
}


def parse_args():
    p = argparse.ArgumentParser(description="Run Excel SIS text training for large/small binary and multi-class with 5-fold 70/10/20 splits.")
    p.add_argument("--excel-root", default=DEFAULT_EXCEL_ROOT, help="Folder containing the four Excel files.")
    p.add_argument("--output-dir", default="/kaggle/working/sis_excel_5fold_fieldaware_binary_mcstrat")
    p.add_argument("--model", default="vinai/phobert-base")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50", help="Threshold sweep written to metrics.json for binary_i63.")
    p.add_argument("--pooling", choices=["cls", "attention", "gated"], default="attention", help="Pooling method after PhoBERT encoder.")
    p.add_argument("--input-mode", choices=["concat", "field"], default="field", help="Input representation mode: concat all fields or encode Excel fields separately.")
    p.add_argument("--max-len-per-field", type=int, default=128)
    p.add_argument("--save-field-attention", action="store_true", help="Save field-level attention weights in field-aware prediction CSVs.")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.2, help="Documented protocol ratio. With 5 folds, test is one fold = 0.2.")
    p.add_argument("--excel-split-label", choices=["target", "binary", "multiclass"], default="multiclass", help="Label source used only to stratify Excel kfold splits.")
    p.add_argument("--only", default="small_binary", help="Comma-separated experiment names to run. Use --only large for all large experiments, --only small for all small experiments, or --only all for every experiment.")
    p.add_argument("--lambda-aux", "--aux-weight", dest="lambda_aux", type=float, default=0.5, help="Auxiliary 3-class loss weight for multitask experiments.")
    p.add_argument("--contrastive-loss", choices=["none", "hard_supcon"], default="none")
    p.add_argument("--contrastive-weight", type=float, default=0.0)
    p.add_argument("--contrastive-temperature", type=float, default=0.1)
    p.add_argument("--hard-negative-weight", type=float, default=2.0)
    p.add_argument("--contrastive-proj-dim", type=int, default=128)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    return p.parse_args()


def selected_experiments(value):
    if not value or value.strip().lower() == "all":
        return EXPERIMENTS
    requested = {item.strip() for item in value.split(",") if item.strip()}
    names = set()
    for item in requested:
        names.update(EXPERIMENT_ALIASES.get(item.lower(), {item}))
    unknown = names - {experiment["name"] for experiment in EXPERIMENTS}
    if unknown:
        aliases = ", ".join(sorted(EXPERIMENT_ALIASES))
        raise ValueError(f"Unknown experiment names: {', '.join(sorted(unknown))}. Available aliases: all, {aliases}")
    return [experiment for experiment in EXPERIMENTS if experiment["name"] in names]


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


def run_stage(name, cmd, done_path, force=False, dry_run=False):
    print("\n" + "=" * 80)
    print(name)
    print(" ".join(str(item) for item in cmd), flush=True)
    if done_path.exists() and not force:
        print(f"Skip: found {done_path}")
        return
    if dry_run:
        return
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if not pythonpath else os.pathsep.join([str(ROOT), pythonpath])
    subprocess.run([str(item) for item in cmd], check=True, cwd=ROOT, env=env)


def train_cmd(args, experiment, fold, out):
    excel_root = args.excel_root.rstrip("/\\")
    data = ",".join(f"{excel_root}/{name}" for name in experiment["data"].split(","))
    input_mode = "concat" if experiment["task"] == "multitask" and args.input_mode == "field" else args.input_mode
    cmd = [
        sys.executable,
        "train_text.py",
        "--data",
        data,
        "--out",
        out,
        "--model",
        args.model,
        "--format",
        "excel",
        "--excel-task",
        experiment["task"],
        "--split-strategy",
        "kfold",
        "--n-folds",
        args.folds,
        "--fold-index",
        fold,
        "--excel-split-label",
        args.excel_split_label,
        "--val-ratio",
        args.val_ratio,
        "--test-ratio",
        args.test_ratio,
        "--binary-positive-label",
        experiment["positive"],
        "--epochs",
        args.epochs,
        "--batch",
        args.batch,
        "--lr",
        args.lr,
        "--wd",
        args.wd,
        "--warmup",
        args.warmup,
        "--max-len",
        args.max_len,
        "--accum",
        args.accum,
        "--seed",
        args.seed,
        "--threshold",
        args.threshold,
        "--thresholds",
        args.thresholds,
        "--pooling",
        args.pooling,
        "--input-mode",
        input_mode,
        "--max-len-per-field",
        args.max_len_per_field,
        "--workers",
        args.workers,
    ]
    if args.save_field_attention:
        cmd.append("--save-field-attention")
    if experiment["task"] == "multitask":
        cmd.extend(["--lambda-aux", args.lambda_aux])
    if args.contrastive_loss != "none" or args.contrastive_weight > 0:
        cmd.extend(
            [
                "--contrastive-loss",
                args.contrastive_loss,
                "--contrastive-weight",
                args.contrastive_weight,
                "--contrastive-temperature",
                args.contrastive_temperature,
                "--hard-negative-weight",
                args.hard_negative_weight,
                "--contrastive-proj-dim",
                args.contrastive_proj_dim,
            ]
        )
    if args.cpu:
        cmd.append("--cpu")
    if args.no_mgpu:
        cmd.append("--no-mgpu")
    return cmd


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.folds != 5 or abs(args.test_ratio - 0.2) > 1e-9:
        print(f"Using {args.folds} folds: held-out test ratio is {1.0 / args.folds:.3f}; requested --test-ratio is recorded as {args.test_ratio}.")
    print(f"Excel kfold stratify label: {args.excel_split_label}")
    if args.input_mode == "field" and any(experiment["task"] == "multitask" for experiment in selected_experiments(args.only)):
        print("Note: multitask experiments use --input-mode concat because field mode is binary/multiclass only.")

    experiments = selected_experiments(args.only)
    for experiment in experiments:
        for fold in range(args.folds):
            out = f"{args.output_dir.rstrip('/\\')}/{experiment['name']}/fold_{fold}"
            run_stage(
                f"{experiment['name']} fold {fold}: train=70%, val=10%, test=20%",
                train_cmd(args, experiment, fold, out),
                Path(out) / "metrics.json",
                force=args.force,
                dry_run=args.dry_run,
            )
        if not args.dry_run:
            aggregate_experiment(output_dir, experiment["name"], args.folds)

    if not args.dry_run:
        print_final_small_report(output_dir, args)


if __name__ == "__main__":
    main()
