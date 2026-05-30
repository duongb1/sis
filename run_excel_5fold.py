import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


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
        "name": "small_multiclass",
        "data": "700_co_label.xlsx,700_khong_label.xlsx",
        "task": "multiclass",
        "positive": "I63_INFARCTION",
    },
]


def parse_args():
    p = argparse.ArgumentParser(description="Run Excel SIS text training for large/small binary and multi-class with 5-fold 70/10/20 splits.")
    p.add_argument("--output-dir", default="/kaggle/working/sis_excel_5fold")
    p.add_argument("--model", default="vinai/phobert-base")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.2, help="Documented protocol ratio. With 5 folds, test is one fold = 0.2.")
    p.add_argument("--only", default=None, help="Comma-separated experiment names to run.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    return p.parse_args()


def selected_experiments(value):
    if not value:
        return EXPERIMENTS
    names = {item.strip() for item in value.split(",") if item.strip()}
    unknown = names - {experiment["name"] for experiment in EXPERIMENTS}
    if unknown:
        raise ValueError(f"Unknown experiment names: {', '.join(sorted(unknown))}")
    return [experiment for experiment in EXPERIMENTS if experiment["name"] in names]


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
    cmd = [
        sys.executable,
        "train_text.py",
        "--data",
        experiment["data"],
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
        "--workers",
        args.workers,
    ]
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

    for experiment in selected_experiments(args.only):
        for fold in range(args.folds):
            out = output_dir / experiment["name"] / f"fold_{fold}"
            run_stage(
                f"{experiment['name']} fold {fold}: train=70%, val=10%, test=20%",
                train_cmd(args, experiment, fold, out),
                out / "metrics.json",
                force=args.force,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
