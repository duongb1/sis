import argparse
import sys
from pathlib import Path

from sislib.reports import aggregate_experiment
from sislib.runner_utils import run_stage


ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Run binary SIS protocol on processed_700.csv and processed_9937.csv.")
    p.add_argument("--small-csv", default="processed_700.csv")
    p.add_argument("--large-csv", default="processed_9937.csv")
    p.add_argument("--output-dir", default="processed_binary_protocol_outputs")
    p.add_argument("--model", default="vinai/phobert-base")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--finetune-epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--finetune-lr", type=float, default=None)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50")
    p.add_argument("--pooling", choices=["cls", "attention", "gated"], default="attention")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    return p.parse_args()


def common_train_args(args, data, out):
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
        "processed",
        "--excel-task",
        "binary",
        "--labels",
        "khong,co",
        "--binary-positive-label",
        "co",
        "--val-ratio",
        args.val_ratio,
        "--test-ratio",
        args.test_ratio,
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
        "concat",
        "--workers",
        args.workers,
    ]
    if args.cpu:
        cmd.append("--cpu")
    if args.no_mgpu:
        cmd.append("--no-mgpu")
    return cmd


def large_cmd(args, out):
    cmd = common_train_args(args, args.large_csv, out)
    cmd.extend(
        [
            "--split-strategy",
            "random",
            "--eval-data",
            f"processed_700_all={args.small_csv}",
            "--eval-format",
            "processed",
            "--eval-split-strategy",
            "eval",
            "--eval-splits",
            "eval",
        ]
    )
    return cmd


def small_fold_cmd(args, fold, out):
    cmd = common_train_args(args, args.small_csv, out)
    cmd.extend(
        [
            "--split-strategy",
            "kfold",
            "--n-folds",
            args.folds,
            "--fold-index",
            fold,
            "--eval-data",
            f"processed_9937_random={args.large_csv}",
            "--eval-format",
            "processed",
            "--eval-split-strategy",
            "random",
            "--eval-splits",
            "val,test",
        ]
    )
    return cmd


def finetune_fold_cmd(args, fold, out, checkpoint):
    cmd = small_fold_cmd(args, fold, out)
    cmd.extend(["--init-checkpoint", checkpoint])
    if args.finetune_epochs is not None:
        idx = cmd.index("--epochs") + 1
        cmd[idx] = args.finetune_epochs
    if args.finetune_lr is not None:
        idx = cmd.index("--lr") + 1
        cmd[idx] = args.finetune_lr
    return cmd


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Check if processed CSVs exist; if not, run preprocess.py automatically
    small_csv_path = Path(args.small_csv)
    if not small_csv_path.exists():
        small_csv_path = ROOT / args.small_csv

    large_csv_path = Path(args.large_csv)
    if not large_csv_path.exists():
        large_csv_path = ROOT / args.large_csv

    if not args.dry_run and (not small_csv_path.exists() or not large_csv_path.exists()):
        # If the user is using the default filenames, try to generate them using preprocess.py
        if args.small_csv == "processed_700.csv" or args.large_csv == "processed_9937.csv":
            preprocess_script = ROOT / "preprocess.py"
            if preprocess_script.exists():
                print("Preprocessed CSV files not found. Running preprocess.py...")
                import subprocess
                subprocess.run([sys.executable, str(preprocess_script)], check=True, cwd=ROOT)
                # Re-check paths after preprocessing
                if not (ROOT / args.small_csv).exists() or not (ROOT / args.large_csv).exists():
                    print("Error: Preprocessing completed but CSV files are still missing.", file=sys.stderr)
                    sys.exit(1)
            else:
                print(f"Error: Preprocessed CSV files not found and {preprocess_script} is missing.", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Error: The specified CSV files do not exist:\n  - Small: {args.small_csv}\n  - Large: {args.large_csv}", file=sys.stderr)
            sys.exit(1)

    print("Protocol:")
    print("model_1 processed_700: 5-fold, train/val/test = 7/1/2, report mean +/- std.")
    print("model_2 processed_9937: random train/val/test = 7/1/2, no 5-fold.")
    print("cross: model_2 evaluates all processed_700; model_1 folds evaluate model_2 val/test split.")
    print("model_3: initialize from model_2 checkpoint, fine-tune each processed_700 fold, test matching fold.")

    large_out = str(output_dir / "model_2_processed_9937_random")
    run_stage(
        "model_2 processed_9937 random 70/10/20 + evaluate all processed_700",
        large_cmd(args, large_out),
        Path(large_out) / "metrics.json",
        force=args.force,
        dry_run=args.dry_run,
        cwd=ROOT,
    )

    for fold in range(args.folds):
        out = str(output_dir / "model_1_processed_700_5fold" / f"fold_{fold}")
        run_stage(
            f"model_1 processed_700 fold {fold}: train=70%, val=10%, test=20% + evaluate processed_9937 val/test",
            small_fold_cmd(args, fold, out),
            Path(out) / "metrics.json",
            force=args.force,
            dry_run=args.dry_run,
            cwd=ROOT,
        )
    if not args.dry_run:
        aggregate_experiment(output_dir, "model_1_processed_700_5fold", args.folds)

    large_checkpoint = str(Path(large_out) / "best_auc_phobert")
    for fold in range(args.folds):
        out = str(output_dir / "model_3_large_to_small_finetune_5fold" / f"fold_{fold}")
        run_stage(
            f"model_3 fine-tune large checkpoint on processed_700 fold {fold}",
            finetune_fold_cmd(args, fold, out, large_checkpoint),
            Path(out) / "metrics.json",
            force=args.force,
            dry_run=args.dry_run,
            cwd=ROOT,
        )
    if not args.dry_run:
        aggregate_experiment(output_dir, "model_3_large_to_small_finetune_5fold", args.folds)


if __name__ == "__main__":
    main()
