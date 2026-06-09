import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run 5-fold MRI 3-class training.")
    parser.add_argument("--folds-csv", default="mri_3class_folds.csv")
    parser.add_argument("--output-dir", default="outputs/mri_3class_5fold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--model", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-images-per-case", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def train_command(args, fold, out):
    cmd = [
        sys.executable,
        "train_mri_3class.py",
        "--folds-csv",
        args.folds_csv,
        "--fold-index",
        str(fold),
        "--out",
        str(out),
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--lr",
        str(args.lr),
        "--wd",
        str(args.wd),
        "--model",
        args.model,
        "--image-size",
        str(args.image_size),
        "--max-images-per-case",
        str(args.max_images_per_case),
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
    ]
    if args.pretrained:
        cmd.append("--pretrained")
    if args.cpu:
        cmd.append("--cpu")
    return cmd


def run_fold(args, fold):
    out = Path(args.output_dir) / f"fold_{fold}"
    metrics_path = out / "metrics.json"
    if metrics_path.exists() and not args.force:
        print(f"skip fold {fold}: {metrics_path} exists")
        return
    cmd = train_command(args, fold, out)
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def read_metric(metrics, path):
    value = metrics
    for key in path:
        value = value[key]
    return value


def write_summary(output_dir, folds):
    rows = []
    for fold in range(folds):
        metrics_path = Path(output_dir) / f"fold_{fold}" / "metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        test = metrics["test"]
        rows.append(
            {
                "fold": fold,
                "test_accuracy": test["accuracy"],
                "test_balanced_accuracy": test["balanced_accuracy"],
                "test_macro_f1": test["macro_f1"],
                "test_weighted_f1": test["weighted_f1"],
                "binary_i63_auc": test["binary_i63"]["auc"],
                "binary_i63_macro_f1": test["binary_i63"]["macro_f1"],
                "binary_i63_sensitivity": test["binary_i63"]["i63_recall_sensitivity"],
                "binary_i63_specificity": test["binary_i63"]["non_i63_recall_specificity"],
            }
        )
    if not rows:
        return
    summary_path = Path(output_dir) / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {summary_path}")


def main():
    args = parse_args()
    if not args.dry_run:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for fold in range(args.folds):
        run_fold(args, fold)
    if not args.dry_run:
        write_summary(args.output_dir, args.folds)


if __name__ == "__main__":
    main()
