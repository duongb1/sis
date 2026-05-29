import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Train/evaluate large and small SIS text folders, then run cross-tests.")
    p.add_argument("--large", default="/kaggle/input/datasets/duongb/cthsis/sis/large")
    p.add_argument("--small", default="/kaggle/input/datasets/duongb/cthsis/sis/small")
    p.add_argument("--output_dir", default="/kaggle/working/sis_runs")
    p.add_argument("--model", default="vinai/phobert-base")
    p.add_argument("--labels", default=None, help="Comma-separated class names. Defaults to union of labels in --large and --small.")
    p.add_argument("--binary-positive-label", default="I63_INFARCTION", help="Class treated as positive for one-vs-rest binary metrics.")
    p.add_argument("--large_text_ckpt", default=None, help="Optional existing large-text checkpoint.")
    p.add_argument("--small_text_ckpt", default=None, help="Optional existing small-text checkpoint.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_text", type=int, default=16)
    p.add_argument("--large_epochs", type=int, default=8)
    p.add_argument("--small_epochs", type=int, default=8)
    p.add_argument("--text_lr", type=float, default=2e-5)
    p.add_argument("--wd_text", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--force", action="store_true", help="Retrain/re-evaluate stages even if outputs already exist.")
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
            "--seed",
            args.seed,
            "--threshold",
            args.threshold,
            "--workers",
            args.workers,
            "--binary-positive-label",
            args.binary_positive_label,
        ]
    )
    if args.cpu:
        cmd.append("--cpu")
    if args.no_mgpu:
        cmd.append("--no-mgpu")
    return cmd


def parse_labels_arg(value):
    if value is None:
        return None
    labels = [item.strip() for item in str(value).split(",") if item.strip()]
    return labels or None


def discover_csv_labels(*roots):
    labels = set()
    for root in roots:
        root = Path(root)
        for split in ("train", "val", "test"):
            split_dir = root / split
            if split_dir.exists():
                labels.update(path.stem for path in split_dir.glob("*.csv") if not path.name.startswith("._"))
                labels.update(path.name for path in split_dir.iterdir() if path.is_dir())
    return sorted(labels)


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
            "--binary-positive-label",
            args.binary_positive_label,
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


def load_confusion_matrix(path):
    path = Path(path)
    if not path.exists():
        return None
    tn = fp = fn = tp = 0
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = int(row["true_label"])
            pred = int(row["pred_label"])
            if label == 0 and pred == 0:
                tn += 1
            elif label == 0 and pred == 1:
                fp += 1
            elif label == 1 and pred == 0:
                fn += 1
            elif label == 1 and pred == 1:
                tp += 1
    return [[tn, fp], [fn, tp]]


def print_metric_summary(root):
    rows = []
    for name, metrics_path, pred_path in [
        ("Large train/test", root / "01_large_text_ce" / "metrics.json", root / "01_large_text_ce" / "test_predictions_best_auc.csv"),
        ("Small train/test", root / "02_small_text_ce" / "metrics.json", root / "02_small_text_ce" / "test_predictions_best_auc.csv"),
        ("Large checkpoint on small test", root / "03_cross_test" / "large_on_small_test" / "metrics.json", root / "03_cross_test" / "large_on_small_test" / "test_predictions.csv"),
        ("Small checkpoint on large test", root / "03_cross_test" / "small_on_large_test" / "metrics.json", root / "03_cross_test" / "small_on_large_test" / "test_predictions.csv"),
    ]:
        metrics = load_json(metrics_path)
        if metrics and "test" in metrics:
            test = metrics["test"]
            rows.append(
                {
                    "model": name,
                    "acc": test.get("accuracy"),
                    "f1": test.get("f1_macro", test.get("f1")),
                    "auc": test.get("auc"),
                    "binary_i63": test.get("binary_i63"),
                    "sens": test.get("sensitivity"),
                    "spec": test.get("specificity"),
                    "cm": test.get("confusion_matrix") or load_confusion_matrix(pred_path),
                }
            )
    print("\n" + "=" * 80)
    print("Summary")
    for row in rows:
        print(
            f"{row['model']}: acc={row['acc']} f1={row['f1']} auc={row['auc']} "
            f"sens={row['sens']} spec={row['spec']}"
        )
        if row["cm"] is not None:
            print(f"  multi-class confusion_matrix: {row['cm']}")
        if row["binary_i63"]:
            binary = row["binary_i63"]
            print(
                f"  binary_i63: acc={binary.get('accuracy')} f1={binary.get('f1')} auc={binary.get('auc')} "
                f"sens={binary.get('sensitivity')} spec={binary.get('specificity')}"
            )
            print(f"  binary_i63 confusion_matrix [[TN, FP], [FN, TP]]: {binary.get('confusion_matrix')}")


def train_text_cmd(args, data, out, epochs, labels):
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
        "--labels",
        ",".join(labels),
    ]
    return add_text_train_flags(cmd, args)


def main():
    args = parse_args()
    root = Path(args.output_dir)
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)

    large_out = root / "01_large_text_ce"
    small_out = root / "02_small_text_ce"
    cross_out = root / "03_cross_test"

    large_ckpt = Path(args.large_text_ckpt) if args.large_text_ckpt else large_out / "best_auc_phobert"
    small_ckpt = Path(args.small_text_ckpt) if args.small_text_ckpt else small_out / "best_auc_phobert"
    labels = parse_labels_arg(args.labels) or discover_csv_labels(args.large, args.small)
    if len(labels) < 2:
        raise RuntimeError(f"Need at least two labels across --large/--small, found: {labels}")
    print(f"Labels ({len(labels)}): {', '.join(labels)}")

    if args.large_text_ckpt:
        print(f"\nUsing existing large-text checkpoint: {large_ckpt}")
    else:
        run_stage(
            "1. Train and evaluate on large",
            train_text_cmd(args, args.large, large_out, args.large_epochs, labels),
            large_ckpt,
            args.force,
            args.dry_run,
        )

    if args.small_text_ckpt:
        print(f"\nUsing existing small-text checkpoint: {small_ckpt}")
    else:
        run_stage(
            "2. Train and evaluate on small",
            train_text_cmd(args, args.small, small_out, args.small_epochs, labels),
            small_ckpt,
            args.force,
            args.dry_run,
        )

    large_on_small_cmd = [
        sys.executable,
        "scripts/eval_text_checkpoint.py",
        "--checkpoint",
        large_ckpt,
        "--dataset",
        "small",
        "--small",
        args.small,
        "--out",
        cross_out / "large_on_small_test",
    ]
    add_text_eval_flags(large_on_small_cmd, args)
    run_stage(
        "3a. Cross-test: large checkpoint on small test",
        large_on_small_cmd,
        cross_out / "large_on_small_test" / "metrics.json",
        args.force,
        args.dry_run,
    )

    small_on_large_cmd = [
        sys.executable,
        "scripts/eval_text_checkpoint.py",
        "--checkpoint",
        small_ckpt,
        "--dataset",
        "large",
        "--texts",
        args.large,
        "--out",
        cross_out / "small_on_large_test",
    ]
    add_text_eval_flags(small_on_large_cmd, args)
    run_stage(
        "3b. Cross-test: small checkpoint on large test",
        small_on_large_cmd,
        cross_out / "small_on_large_test" / "metrics.json",
        args.force,
        args.dry_run,
    )

    if not args.dry_run:
        print_metric_summary(root)


if __name__ == "__main__":
    main()
