import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_KAGGLE_MRI_ROOT = "images"


def parse_args():
    parser = argparse.ArgumentParser(description="Run 5-fold MRI binary training.")
    parser.add_argument("--folds-csv", default="")
    parser.add_argument("--image-root", default="images")
    parser.add_argument("--output-dir", default="mri_binary_5fold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-images-per-case", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-dp", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def train_command(args, fold, out):
    cmd = [
        sys.executable,
        "train_mri_3class.py",
        "--folds-csv",
        args.folds_csv,
        "--image-root",
        args.image_root,
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
        "--image-size",
        str(args.image_size),
        "--max-images-per-case",
        str(args.max_images_per_case),
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        "--precision",
        args.precision,
    ]
    if args.pretrained:
        cmd.append("--pretrained")
    if args.cpu:
        cmd.append("--cpu")
    if args.no_dp:
        cmd.append("--no-dp")
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
                "test_accuracy": test.get("accuracy"),
                "test_f1": test.get("f1"),
                "test_auc": test.get("auc"),
                "test_sensitivity": test.get("sensitivity"),
                "test_specificity": test.get("specificity"),
                "test_brier_score": test.get("brier_score"),
                "test_ece": test.get("ece"),
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
