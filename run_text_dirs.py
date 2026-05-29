import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Train text-only PhoBERT checkpoints for large and small SIS text folders.")
    p.add_argument("--large", default="/kaggle/input/datasets/duongb/cthsis/sis/large", help="Large text folder. Supports split/label/*.txt or split/{co,khong}.csv.")
    p.add_argument("--small", default="/kaggle/input/datasets/duongb/cthsis/sis/small", help="Small text folder. Supports split/label/*.txt or split/{co,khong}.csv.")
    p.add_argument("--output_dir", default="/kaggle/working/sis_runs_text_dirs")
    p.add_argument("--model", default="vinai/phobert-base")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--large_epochs", type=int, default=None)
    p.add_argument("--small_epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--force", action="store_true", help="Retrain even if best checkpoint already exists.")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def run_stage(name, cmd, done_path, force, dry_run):
    print("\n" + "=" * 80)
    print(name)
    print(" ".join(str(x) for x in cmd), flush=True)
    if done_path.exists() and not force:
        print(f"Skip: found {done_path}")
        return
    if dry_run:
        return
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if not pythonpath else os.pathsep.join([str(ROOT), pythonpath])
    subprocess.run([str(x) for x in cmd], check=True, cwd=ROOT, env=env)


def train_cmd(args, data, out, epochs):
    cmd = [
        sys.executable,
        "train_text.py",
        "--data",
        data,
        "--out",
        out,
        "--model",
        args.model,
        "--epochs",
        epochs,
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
    out = Path(args.output_dir)
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)

    large_out = out / "01_large_text_ce"
    small_out = out / "02_small_text_ce"

    run_stage(
        "1. Large text-only CE",
        train_cmd(args, args.large, large_out, args.large_epochs or args.epochs),
        large_out / "best_auc_phobert",
        args.force,
        args.dry_run,
    )
    run_stage(
        "2. Small text-only CE",
        train_cmd(args, args.small, small_out, args.small_epochs or args.epochs),
        small_out / "best_auc_phobert",
        args.force,
        args.dry_run,
    )


if __name__ == "__main__":
    main()
