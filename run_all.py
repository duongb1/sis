import argparse
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Run the stacking meta-classifier pipeline.")
    p.add_argument("--images", default=None)
    p.add_argument("--paired_train_csv", default=None)
    p.add_argument("--paired_val_csv", default=None)
    p.add_argument("--paired_test_csv", default=None)
    p.add_argument("--large_text_ckpt", required=True)
    p.add_argument("--paired_text_model_name_or_ckpt", default="vinai/phobert-base")
    p.add_argument("--mri_teacher_pred_csv", default=None)
    p.add_argument("--mri_teacher_dir", default=None)
    p.add_argument("--mri_teacher_ckpt", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--batch_mri", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no_mgpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cmd = [
        sys.executable,
        "scripts/run_stacking_meta_classifier.py",
        "--large_text_ckpt", args.large_text_ckpt,
        "--paired_text_model_name_or_ckpt", args.paired_text_model_name_or_ckpt,
        "--output_dir", args.output_dir,
        "--n_folds", str(args.n_folds),
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--batch_mri", str(args.batch_mri),
        "--lr", str(args.lr),
        "--wd", str(args.wd),
        "--warmup", str(args.warmup),
        "--max_len", str(args.max_len),
        "--threshold", str(args.threshold),
        "--workers", str(args.workers),
        "--accum", str(args.accum),
    ]
    if args.images:
        cmd.extend(["--images", args.images])
    if args.paired_train_csv:
        cmd.extend(["--paired_train_csv", args.paired_train_csv])
    if args.paired_val_csv:
        cmd.extend(["--paired_val_csv", args.paired_val_csv])
    if args.paired_test_csv:
        cmd.extend(["--paired_test_csv", args.paired_test_csv])
    if args.mri_teacher_pred_csv:
        cmd.extend(["--mri_teacher_pred_csv", args.mri_teacher_pred_csv])
    if args.mri_teacher_dir:
        cmd.extend(["--mri_teacher_dir", args.mri_teacher_dir])
    if args.mri_teacher_ckpt:
        cmd.extend(["--mri_teacher_ckpt", args.mri_teacher_ckpt])
    if args.cpu:
        cmd.append("--cpu")
    if args.no_mgpu:
        cmd.append("--no_mgpu")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
