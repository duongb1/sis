import argparse
import csv
import subprocess
import sys
from pathlib import Path

from run_excel_5fold import collect_model_report, mean_std_text


ROOT = Path(__file__).resolve().parent


VARIANTS = [
    {
        "name": "small_gated_mtl_aux_0_5",
        "experiment": "small_multitask",
        "metrics_folder": "small_multitask",
        "lambda_aux": 0.5,
    },
    {
        "name": "large_gated_mtl_aux_0_5",
        "experiment": "large_multitask",
        "metrics_folder": "large_multitask",
        "lambda_aux": 0.5,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare small and large PhoBERT gated-fusion multitask models.")
    parser.add_argument("--excel-root", default="/kaggle/input/datasets/duongbui/siscth")
    parser.add_argument("--output-dir", default="/kaggle/working/sis_excel_5fold_gated_mtl_compare_mcstrat")
    parser.add_argument("--model", default="vinai/phobert-base")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16, help="Per-step batch. DataParallel splits this across both T4 GPUs.")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--excel-split-label", choices=["target", "binary", "multiclass"], default="multiclass")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-mgpu", action="store_true", help="Disable DataParallel. Leave unset on Kaggle dual T4.")
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
        variant["experiment"],
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
        "gated",
        "--input-mode",
        "concat",
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
        "--lambda-aux",
        variant["lambda_aux"],
    ]
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


def write_summary(output_dir, reports):
    path = output_dir / "summary_compare.csv"
    fields = [
        "model",
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
        for model_name, report in reports.items():
            summary = report["summary"]
            counts = report["counts"]
            writer.writerow(
                {
                    "model": model_name,
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


def print_table(reports):
    print("\nModel                              Acc       F1        AUC       Sens      Spec      BalAcc    TN/FP/FN/TP")
    for model_name, report in reports.items():
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
            f"{counts['tn']}/{counts['fp']}/{counts['fn']}/{counts['tp']}"
        )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("Gated fusion comparison:", flush=True)
    print("- small_gated_mtl_aux_0_5", flush=True)
    print("- large_gated_mtl_aux_0_5", flush=True)

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
        report = collect_model_report(output_dir / variant["name"], variant["metrics_folder"], args.folds)
        if report is None:
            raise FileNotFoundError(f"Missing fold metrics for {variant['name']}")
        reports[variant["name"]] = report

    summary_path = write_summary(output_dir, reports)
    print_table(reports)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
