import argparse
import csv
import subprocess
import sys
from pathlib import Path

from run_excel_5fold import collect_model_report, mean_std_text


ROOT = Path(__file__).resolve().parent


VARIANTS = [
    {
        "name": "small_binary_attnpool",
        "contrastive_loss": "none",
        "contrastive_weight": 0.0,
    },
    {
        "name": "small_attnpool_hard_supcon_0_1",
        "contrastive_loss": "hard_supcon",
        "contrastive_weight": 0.1,
    },
    {
        "name": "small_attnpool_hard_supcon_0_2",
        "contrastive_loss": "hard_supcon",
        "contrastive_weight": 0.2,
    },
    {
        "name": "small_attnpool_hard_supcon_0_3",
        "contrastive_loss": "hard_supcon",
        "contrastive_weight": 0.3,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare small binary attention pooling against hard-negative SupCon variants.")
    parser.add_argument("--excel-root", default="/kaggle/input/datasets/duongbui/siscth")
    parser.add_argument("--output-dir", default="/kaggle/working/sis_excel_5fold_hard_supcon_mcstrat")
    parser.add_argument("--model", default="vinai/phobert-base")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--max-len-per-field", type=int, default=128)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--excel-split-label", choices=["target", "binary", "multiclass"], default="multiclass")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--hard-negative-weight", type=float, default=2.0)
    parser.add_argument("--contrastive-proj-dim", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-mgpu", action="store_true")
    return parser.parse_args()


def build_command(args, variant):
    cmd = [
        sys.executable,
        "run_excel_5fold.py",
        "--excel-root",
        args.excel_root,
        "--output-dir",
        str(Path(args.output_dir) / variant["name"]),
        "--model",
        args.model,
        "--only",
        "small_binary",
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
        "attention",
        "--input-mode",
        "concat",
        "--max-len-per-field",
        args.max_len_per_field,
        "--workers",
        args.workers,
        "--folds",
        args.folds,
        "--val-ratio",
        args.val_ratio,
        "--test-ratio",
        args.test_ratio,
        "--excel-split-label",
        args.excel_split_label,
    ]
    if variant["contrastive_loss"] != "none":
        cmd.extend(
            [
                "--contrastive-loss",
                variant["contrastive_loss"],
                "--contrastive-weight",
                variant["contrastive_weight"],
                "--contrastive-temperature",
                args.contrastive_temperature,
                "--hard-negative-weight",
                args.hard_negative_weight,
                "--contrastive-proj-dim",
                args.contrastive_proj_dim,
            ]
        )
    if args.force:
        cmd.append("--force")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.cpu:
        cmd.append("--cpu")
    if args.no_mgpu:
        cmd.append("--no-mgpu")
    return [str(item) for item in cmd]


def summary_value(summary, key, stat):
    value = summary.get(key, {}).get(stat)
    return "" if value is None else f"{value:.6f}"


def write_compare_summary(output_dir, reports):
    path = output_dir / "summary_compare.csv"
    fields = [
        "model",
        "contrastive_loss",
        "contrastive_weight",
        "accuracy_mean",
        "accuracy_std",
        "f1_mean",
        "f1_std",
        "auc_mean",
        "auc_std",
        "sensitivity_mean",
        "sensitivity_std",
        "specificity_mean",
        "specificity_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "TN",
        "FP",
        "FN",
        "TP",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model_name, (variant, report) in reports.items():
            summary = report["summary"]
            counts = report["counts"]
            writer.writerow(
                {
                    "model": model_name,
                    "contrastive_loss": variant["contrastive_loss"],
                    "contrastive_weight": f"{variant['contrastive_weight']:g}",
                    "accuracy_mean": summary_value(summary, "test.accuracy", "mean"),
                    "accuracy_std": summary_value(summary, "test.accuracy", "std"),
                    "f1_mean": summary_value(summary, "test.f1", "mean"),
                    "f1_std": summary_value(summary, "test.f1", "std"),
                    "auc_mean": summary_value(summary, "test.auc", "mean"),
                    "auc_std": summary_value(summary, "test.auc", "std"),
                    "sensitivity_mean": summary_value(summary, "test.sensitivity", "mean"),
                    "sensitivity_std": summary_value(summary, "test.sensitivity", "std"),
                    "specificity_mean": summary_value(summary, "test.specificity", "mean"),
                    "specificity_std": summary_value(summary, "test.specificity", "std"),
                    "balanced_accuracy_mean": summary_value(summary, "test.balanced_accuracy", "mean"),
                    "balanced_accuracy_std": summary_value(summary, "test.balanced_accuracy", "std"),
                    "TN": counts["tn"],
                    "FP": counts["fp"],
                    "FN": counts["fn"],
                    "TP": counts["tp"],
                }
            )
    return path


def metric_mean(report, key):
    return report["summary"].get(key, {}).get("mean")


def print_compare_table(reports):
    print("\nModel                              Acc       F1        AUC       Sens      Spec      BalAcc    FP/FN")
    for model_name, (variant, report) in reports.items():
        summary = report["summary"]
        counts = report["counts"]
        print(
            f"{model_name:<34} "
            f"{mean_std_text(summary, 'test.accuracy'):<9} "
            f"{mean_std_text(summary, 'test.f1'):<9} "
            f"{mean_std_text(summary, 'test.auc'):<9} "
            f"{mean_std_text(summary, 'test.sensitivity'):<9} "
            f"{mean_std_text(summary, 'test.specificity'):<9} "
            f"{mean_std_text(summary, 'test.balanced_accuracy'):<9} "
            f"{counts['fp']}/{counts['fn']}"
        )


def print_baseline_deltas(reports):
    baseline = reports.get("small_binary_attnpool")
    if not baseline:
        return
    baseline_report = baseline[1]
    baseline_counts = baseline_report["counts"]
    print("\nFP/FN trade-off vs small_binary_attnpool:")
    for model_name, (variant, report) in reports.items():
        if model_name == "small_binary_attnpool":
            continue
        counts = report["counts"]
        delta_fp = counts["fp"] - baseline_counts["fp"]
        delta_fn = counts["fn"] - baseline_counts["fn"]
        delta_auc = metric_mean(report, "test.auc") - metric_mean(baseline_report, "test.auc")
        delta_f1 = metric_mean(report, "test.f1") - metric_mean(baseline_report, "test.f1")
        delta_sens = metric_mean(report, "test.sensitivity") - metric_mean(baseline_report, "test.sensitivity")
        delta_spec = metric_mean(report, "test.specificity") - metric_mean(baseline_report, "test.specificity")
        delta_bal = metric_mean(report, "test.balanced_accuracy") - metric_mean(baseline_report, "test.balanced_accuracy")
        print(
            f"{model_name}: "
            f"dFP={delta_fp:+d}, dFN={delta_fn:+d}, "
            f"dAUC={delta_auc:+.3f}, dF1={delta_f1:+.3f}, "
            f"dSens={delta_sens:+.3f}, dSpec={delta_spec:+.3f}, dBalAcc={delta_bal:+.3f}"
        )


def best_by_metric(reports, metric_key, reverse=True):
    values = []
    for model_name, (variant, report) in reports.items():
        value = metric_mean(report, metric_key)
        if value is not None:
            values.append((value, model_name))
    if not values:
        return "n/a"
    values.sort(reverse=reverse)
    return values[0][1]


def best_by_count(reports, count_key):
    values = [(report["counts"][count_key], model_name) for model_name, (variant, report) in reports.items()]
    values.sort()
    return values[0][1] if values else "n/a"


def print_best_models(reports):
    print("\nBest models:")
    print(f"best_f1: {best_by_metric(reports, 'test.f1')}")
    print(f"best_auc: {best_by_metric(reports, 'test.auc')}")
    print(f"best_sensitivity: {best_by_metric(reports, 'test.sensitivity')}")
    print(f"best_specificity: {best_by_metric(reports, 'test.specificity')}")
    print(f"best_balanced_accuracy: {best_by_metric(reports, 'test.balanced_accuracy')}")
    print(f"lowest_fn: {best_by_count(reports, 'fn')}")
    print(f"lowest_fp: {best_by_count(reports, 'fp')}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("Hard-negative SupCon comparison:", flush=True)
    for variant in VARIANTS:
        print(f"- {variant['name']}", flush=True)

    for variant in VARIANTS:
        print("\n" + "=" * 80, flush=True)
        print(f"Running {variant['name']}", flush=True)
        cmd = build_command(args, variant)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)

    if args.dry_run:
        return

    reports = {}
    for variant in VARIANTS:
        report = collect_model_report(output_dir / variant["name"], "small_binary", args.folds)
        if report is None:
            raise FileNotFoundError(f"Missing fold metrics for {variant['name']}")
        reports[variant["name"]] = (variant, report)

    summary_path = write_compare_summary(output_dir, reports)
    print_compare_table(reports)
    print_baseline_deltas(reports)
    print_best_models(reports)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
