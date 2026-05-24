import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Run the full SIS experiment pipeline.")
    p.add_argument("--images", default="/kaggle/input/datasets/duongb/cthsis/images")
    p.add_argument("--texts", default="/kaggle/input/datasets/duongb/cthsis/texts")
    p.add_argument("--output_dir", default="/kaggle/working/sis_runs")
    p.add_argument("--model", default="vinai/phobert-base")
    p.add_argument("--large_text_ckpt", default=None, help="Optional existing large-text checkpoint.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_text", type=int, default=16)
    p.add_argument("--batch_mri", type=int, default=64)
    p.add_argument("--mri_epochs", type=int, default=20)
    p.add_argument("--large_epochs", type=int, default=8)
    p.add_argument("--paired_epochs", type=int, default=8)
    p.add_argument("--mri_lr", type=float, default=1e-4)
    p.add_argument("--text_lr", type=float, default=2e-5)
    p.add_argument("--wd_mri", type=float, default=1e-4)
    p.add_argument("--wd_text", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--force", action="store_true", help="Retrain stages even if outputs already exist.")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no_mgpu", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def run_stage(name, cmd, done_path, force=False, dry_run=False):
    print("\n" + "=" * 80)
    print(name)
    print(" ".join(str(x) for x in cmd), flush=True)
    if done_path and Path(done_path).exists() and not force:
        print(f"Skip: found {done_path}")
        return
    if dry_run:
        return
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if not pythonpath else os.pathsep.join([str(ROOT), pythonpath])
    subprocess.run([str(x) for x in cmd], check=True, cwd=ROOT, env=env)


def add_common_flags(cmd, args):
    cmd.extend(["--seed", args.seed, "--threshold", args.threshold, "--workers", args.workers])
    if args.cpu:
        cmd.append("--cpu")
    if args.no_mgpu:
        cmd.append("--no-mgpu")
    return cmd


def add_text_train_flags(cmd, args):
    cmd.extend(
        [
            "--batch",
            args.batch_text,
            "--lr",
            args.text_lr,
            "--wd",
            args.wd_text,
            "--warmup",
            args.warmup,
            "--max-len",
            args.max_len,
            "--accum",
            args.accum,
        ]
    )
    return add_common_flags(cmd, args)


def add_text_eval_flags(cmd, args):
    cmd.extend(
        [
            "--seed",
            args.seed,
            "--batch",
            args.batch_text,
            "--max-len",
            args.max_len,
            "--threshold",
            args.threshold,
            "--workers",
            args.workers,
        ]
    )
    if args.cpu:
        cmd.append("--cpu")
    if args.no_mgpu:
        cmd.append("--no-mgpu")
    return cmd


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_metric_summary(root):
    rows = []
    for name, metrics_path in [
        ("MRI-only teacher", root / "00_mri_teacher" / "metrics.json"),
        ("Large-text CE", root / "01_large_text_ce" / "metrics.json"),
        ("Paired-only CE", root / "02_paired_text_ce" / "metrics.json"),
        ("Large checkpoint on paired test", root / "03_cross_test" / "large_on_paired_test" / "metrics.json"),
        ("Paired checkpoint on large test", root / "03_cross_test" / "paired_on_large_test" / "metrics.json"),
    ]:
        metrics = load_json(metrics_path)
        if metrics and "test" in metrics:
            test = metrics["test"]
            rows.append(
                {
                    "model": name,
                    "acc": test.get("accuracy"),
                    "f1": test.get("f1"),
                    "auc": test.get("auc"),
                    "sens": test.get("sensitivity"),
                    "spec": test.get("specificity"),
                }
            )
    print("\n" + "=" * 80)
    print("Summary")
    for row in rows:
        print(
            f"{row['model']}: acc={row['acc']} f1={row['f1']} auc={row['auc']} "
            f"sens={row['sens']} spec={row['spec']}"
        )


def main():
    args = parse_args()
    root = Path(args.output_dir)
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)

    mri_out = root / "00_mri_teacher"
    large_out = root / "01_large_text_ce"
    paired_out = root / "02_paired_text_ce"
    cross_out = root / "03_cross_test"

    mri_ckpt = mri_out / "best_auc_model.pt"
    large_ckpt = Path(args.large_text_ckpt) if args.large_text_ckpt else large_out / "best_auc_phobert"
    paired_ckpt = paired_out / "best_auc_phobert"

    mri_cmd = [
        sys.executable,
        "train_mri.py",
        "--images",
        args.images,
        "--out",
        mri_out,
        "--epochs",
        args.mri_epochs,
        "--batch",
        args.batch_mri,
        "--lr",
        args.mri_lr,
        "--wd",
        args.wd_mri,
    ]
    add_common_flags(mri_cmd, args)
    run_stage("1. MRI-only teacher", mri_cmd, mri_ckpt, args.force, args.dry_run)

    if args.large_text_ckpt:
        print(f"\nUsing existing large-text checkpoint: {large_ckpt}")
    else:
        large_cmd = [
            sys.executable,
            "train_text.py",
            "--data",
            args.texts,
            "--out",
            large_out,
            "--model",
            args.model,
            "--epochs",
            args.large_epochs,
        ]
        add_text_train_flags(large_cmd, args)
        run_stage("2. Large text-only CE", large_cmd, large_ckpt, args.force, args.dry_run)

    paired_cmd = [
        sys.executable,
        "train_pair_text.py",
        "--images",
        args.images,
        "--out",
        paired_out,
        "--model",
        args.model,
        "--epochs",
        args.paired_epochs,
    ]
    add_text_train_flags(paired_cmd, args)
    run_stage("3. Paired text-only CE", paired_cmd, paired_ckpt, args.force, args.dry_run)

    large_on_paired_cmd = [
        sys.executable,
        "scripts/eval_text_checkpoint.py",
        "--checkpoint",
        large_ckpt,
        "--dataset",
        "paired",
        "--images",
        args.images,
        "--out",
        cross_out / "large_on_paired_test",
    ]
    add_text_eval_flags(large_on_paired_cmd, args)
    run_stage(
        "4. Cross-test: large checkpoint on paired test",
        large_on_paired_cmd,
        cross_out / "large_on_paired_test" / "metrics.json",
        args.force,
        args.dry_run,
    )

    paired_on_large_cmd = [
        sys.executable,
        "scripts/eval_text_checkpoint.py",
        "--checkpoint",
        paired_ckpt,
        "--dataset",
        "large",
        "--texts",
        args.texts,
        "--out",
        cross_out / "paired_on_large_test",
    ]
    add_text_eval_flags(paired_on_large_cmd, args)
    run_stage(
        "5. Cross-test: paired checkpoint on large test",
        paired_on_large_cmd,
        cross_out / "paired_on_large_test" / "metrics.json",
        args.force,
        args.dry_run,
    )

    if not args.dry_run:
        print_metric_summary(root)


if __name__ == "__main__":
    main()
