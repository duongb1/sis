import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

VARIANTS = ["cls", "attention", "gated"]


DEFAULT_SMALL_CSV = "/kaggle/input/datasets/duongbui/siscth/small.csv" if Path("/kaggle/input/datasets/duongbui/siscth/small.csv").exists() else "data/small.csv"
DEFAULT_LARGE_CSV = "/kaggle/input/datasets/duongbui/siscth/large.csv" if Path("/kaggle/input/datasets/duongbui/siscth/large.csv").exists() else "data/large.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PhoBERT binary classification models directly on raw small.csv and large.csv.")
    parser.add_argument("--scale", choices=["small", "large"], default="small", help="Dataset scale to run.")
    parser.add_argument("--small-csv", default=DEFAULT_SMALL_CSV, help="Path to small dataset CSV.")
    parser.add_argument("--large-csv", default=DEFAULT_LARGE_CSV, help="Path to large dataset CSV.")
    parser.add_argument("--output-dir", default="processed_compare_outputs", help="Output directory for metrics and checkpoints.")
    parser.add_argument("--model", default="vinai/phobert-base")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5, help="Number of folds (only used for small scale).")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--force", action="store_true", help="Force overwrite existing runs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-mgpu", action="store_true")
    return parser.parse_args()


def load_json(path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_binary_metrics(data):
    if not data:
        return {}
    test = data.get("test", {})
    if "binary_i63" in test:
        return dict(test["binary_i63"])
    return dict(test)


def confusion_counts(metrics):
    if "confusion_matrix" in metrics:
        (tn, fp), (fn, tp) = metrics["confusion_matrix"]
        return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return {key: int(metrics.get(key, 0)) for key in ("tn", "fp", "fn", "tp")}


def summarize_metric_rows(rows):
    keys = sorted(set().union(*(row.keys() for row in rows)))
    summary = {}
    for key in keys:
        values = np.array([row[key] for row in rows if key in row and row[key] is not None and not np.isnan(row[key])], dtype=np.float64)
        if values.size == 0:
            continue
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        }
    return summary


def collect_results(output_dir, variant_folder, is_kfold, folds=5):
    if is_kfold:
        rows = []
        counts = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
        for fold in range(folds):
            path = output_dir / variant_folder / f"fold_{fold}" / "metrics.json"
            data = load_json(path)
            if data:
                metrics = get_binary_metrics(data)
                rows.append(metrics)
                fold_counts = confusion_counts(metrics)
                for key in counts:
                    counts[key] += fold_counts[key]
        if not rows:
            return None
        return {"summary": summarize_metric_rows(rows), "counts": counts}
    else:
        path = output_dir / variant_folder / "metrics.json"
        data = load_json(path)
        if not data:
            return None
        metrics = get_binary_metrics(data)
        summary = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                summary[key] = {"mean": float(value), "std": 0.0}
        counts = confusion_counts(metrics)
        return {"summary": summary, "counts": counts}


def run_stage(desc, cmd, force=False, dry_run=False):
    print("\n" + "=" * 80)
    print(desc)
    print("Command:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    csv_name = args.small_csv if args.scale == "small" else args.large_csv
    csv_path = Path(csv_name)
    if not csv_path.exists():
        csv_path = ROOT / csv_name

    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    is_kfold = (args.scale == "small")

    # Run variants
    for pooling in VARIANTS:
        folder_name = f"{args.scale}_binary_{pooling}"
        if is_kfold:
            for fold in range(args.folds):
                out = output_dir / folder_name / f"fold_{fold}"
                metrics_path = out / "metrics.json"
                if metrics_path.exists() and not args.force:
                    print(f"Skip fold {fold}: {metrics_path} exists")
                    continue
                cmd = [
                    sys.executable, "train_text.py",
                    "--data", str(csv_path),
                    "--out", str(out),
                    "--model", args.model,
                    "--format", "csv",
                    "--excel-task", "binary",
                    "--labels", "khong,co",
                    "--binary-positive-label", "co",
                    "--split-strategy", "kfold",
                    "--n-folds", str(args.folds),
                    "--fold-index", str(fold),
                    "--val-ratio", str(args.val_ratio),
                    "--test-ratio", str(args.test_ratio),
                    "--epochs", str(args.epochs),
                    "--batch", str(args.batch),
                    "--lr", str(args.lr),
                    "--wd", str(args.wd),
                    "--warmup", str(args.warmup),
                    "--max-len", str(args.max_len),
                    "--accum", str(args.accum),
                    "--seed", str(args.seed),
                    "--threshold", str(args.threshold),
                    "--pooling", pooling,
                    "--input-mode", "concat",
                    "--workers", str(args.workers),
                ]
                if args.cpu:
                    cmd.append("--cpu")
                if args.no_mgpu:
                    cmd.append("--no-mgpu")
                run_stage(f"Training {folder_name} fold {fold}", cmd, force=args.force, dry_run=args.dry_run)
            
            # Aggregate folds
            if not args.dry_run:
                from utils.metrics import aggregate_experiment
                aggregate_experiment(output_dir, folder_name, args.folds)
        else:
            out = output_dir / folder_name
            metrics_path = out / "metrics.json"
            if metrics_path.exists() and not args.force:
                print(f"Skip: {metrics_path} exists")
                continue
            cmd = [
                sys.executable, "train_text.py",
                "--data", str(csv_path),
                "--out", str(out),
                "--model", args.model,
                "--format", "csv",
                "--excel-task", "binary",
                "--labels", "khong,co",
                "--binary-positive-label", "co",
                "--split-strategy", "random",
                "--val-ratio", str(args.val_ratio),
                "--test-ratio", str(args.test_ratio),
                "--epochs", str(args.epochs),
                "--batch", str(args.batch),
                "--lr", str(args.lr),
                "--wd", str(args.wd),
                "--warmup", str(args.warmup),
                "--max-len", str(args.max_len),
                "--accum", str(args.accum),
                "--seed", str(args.seed),
                "--threshold", str(args.threshold),
                "--pooling", pooling,
                "--input-mode", "concat",
                "--workers", str(args.workers),
            ]
            if args.cpu:
                cmd.append("--cpu")
            if args.no_mgpu:
                cmd.append("--no-mgpu")
            run_stage(f"Training {folder_name} (Single Split 70/10/20)", cmd, force=args.force, dry_run=args.dry_run)

    if args.dry_run:
        return

    # Load results
    reports = {}
    for pooling in VARIANTS:
        folder_name = f"{args.scale}_binary_{pooling}"
        report = collect_results(output_dir, folder_name, is_kfold, args.folds)
        if report:
            reports[pooling] = report

    if not reports:
        print("\nNo metrics files found to compile summary.")
        return

    # Write CSV summary
    summary_path = output_dir / f"{args.scale}_summary_compare.csv"
    fields = [
        "pooling",
        "accuracy_mean", "accuracy_std",
        "f1_mean", "f1_std",
        "auc_mean", "auc_std",
        "sensitivity_mean", "sensitivity_std",
        "specificity_mean", "specificity_std",
        "brier_score_mean", "brier_score_std",
        "ece_mean", "ece_std",
        "TN", "FP", "FN", "TP"
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for pooling, report in reports.items():
            summary = report["summary"]
            counts = report["counts"]
            
            def get_val(key, stat):
                return summary.get(key, {}).get(stat, 0.0)
                
            writer.writerow({
                "pooling": pooling,
                "accuracy_mean": f"{get_val('accuracy', 'mean'):.6f}",
                "accuracy_std": f"{get_val('accuracy', 'std'):.6f}",
                "f1_mean": f"{get_val('f1', 'mean'):.6f}",
                "f1_std": f"{get_val('f1', 'std'):.6f}",
                "auc_mean": f"{get_val('auc', 'mean'):.6f}",
                "auc_std": f"{get_val('auc', 'std'):.6f}",
                "sensitivity_mean": f"{get_val('sensitivity', 'mean'):.6f}",
                "sensitivity_std": f"{get_val('sensitivity', 'std'):.6f}",
                "specificity_mean": f"{get_val('specificity', 'mean'):.6f}",
                "specificity_std": f"{get_val('specificity', 'std'):.6f}",
                "brier_score_mean": f"{get_val('brier_score', 'mean'):.6f}",
                "brier_score_std": f"{get_val('brier_score', 'std'):.6f}",
                "ece_mean": f"{get_val('ece', 'mean'):.6f}",
                "ece_std": f"{get_val('ece', 'std'):.6f}",
                "TN": counts["tn"],
                "FP": counts["fp"],
                "FN": counts["fn"],
                "TP": counts["tp"],
            })

    # Print summary table
    print("\n" + "=" * 100)
    if is_kfold:
        print(f"5-fold summary: {args.scale} models (mean ± std)")
    else:
        print(f"Summary: {args.scale} models (Single Random Split)")
    print("=" * 100)
    print(f"{'Pooling':10} {'Acc':12} {'F1':12} {'AUC':12} {'Sens':12} {'Spec':12} {'Brier':12} {'ECE':12} FP/FN")
    
    for pooling, report in reports.items():
        summary = report["summary"]
        counts = report["counts"]
        
        def format_metric(key):
            stats = summary.get(key)
            if not stats:
                return "n/a"
            if is_kfold:
                return f"{stats['mean']:.3f}±{stats['std']:.3f}"
            else:
                return f"{stats['mean']:.3f}"

        print(
            f"{pooling:10} "
            f"{format_metric('accuracy'):12} "
            f"{format_metric('f1'):12} "
            f"{format_metric('auc'):12} "
            f"{format_metric('sensitivity'):12} "
            f"{format_metric('specificity'):12} "
            f"{format_metric('brier_score'):12} "
            f"{format_metric('ece'):12} "
            f"{counts['fp']}/{counts['fn']}"
        )
    print(f"\nWrote summary comparing table to {summary_path}\n")


if __name__ == "__main__":
    main()
