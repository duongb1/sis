import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run the full synchronized SIS experiment pipeline once.")
    p.add_argument("--images", default="/kaggle/input/datasets/duongb/cthsis/images")
    p.add_argument("--texts", default="/kaggle/input/datasets/duongb/cthsis/texts")
    p.add_argument("--out-root", default="/kaggle/working/sis_runs")
    p.add_argument("--base-model", default="vinai/phobert-base")

    p.add_argument("--mri-epochs", type=int, default=20)
    p.add_argument("--mri-lr", type=float, default=1e-4)
    p.add_argument("--mri-batch", type=int, default=64)

    p.add_argument("--large-epochs", type=int, default=8)
    p.add_argument("--large-lr", type=float, default=2e-5)
    p.add_argument("--large-batch", type=int, default=16)

    p.add_argument("--paired-epochs", type=int, default=8)
    p.add_argument("--paired-lr", type=float, default=2e-5)
    p.add_argument("--paired-batch", type=int, default=16)

    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--wd-text", type=float, default=0.01)
    p.add_argument("--wd-mri", type=float, default=1e-4)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--accum", type=int, default=1)

    p.add_argument("--kd-alpha", type=float, default=0.05)
    p.add_argument("--kd-loss", choices=["binary", "kl"], default="binary")
    p.add_argument("--kd-temp", type=float, default=2.0)
    p.add_argument("--lupi-alpha", type=float, default=0.2)
    p.add_argument("--lupi-weight-min", type=float, default=0.75)
    p.add_argument("--lupi-weight-max", type=float, default=1.25)

    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def add_flag(cmd, enabled, flag):
    if enabled:
        cmd.append(flag)


def run_stage(name, cmd, dry_run):
    print("\n" + "=" * 80)
    print(name)
    print(" ".join(cmd))
    print("=" * 80, flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def read_split_metrics(run_dir, split):
    path = run_dir / "metrics.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        metrics = json.load(f)
    if split in metrics:
        return metrics[split]
    if metrics:
        return next(iter(metrics.values()))
    return None


def read_confusion_matrix(run_dir, split):
    path = run_dir / f"{split}_predictions_best_auc.csv"
    if not path.exists():
        return None
    tn = fp = fn = tp = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            y = int(row["true_label"])
            pred = int(row["pred_label"])
            if y == 0 and pred == 0:
                tn += 1
            elif y == 0 and pred == 1:
                fp += 1
            elif y == 1 and pred == 0:
                fn += 1
            elif y == 1 and pred == 1:
                tp += 1
    return [[tn, fp], [fn, tp]]


def summarize_results(out_root, rows):
    summary = []
    for row in rows:
        metrics = read_split_metrics(row["dir"], row["split"])
        cm = read_confusion_matrix(row["dir"], row["split"])
        item = {k: row[k] for k in ["model", "init", "train_data", "mri_used", "loss", "test"]}
        item["metrics_file"] = str(row["dir"] / "metrics.json")
        item["confusion_matrix"] = cm
        if metrics is None:
            item.update({"accuracy": None, "f1": None, "auc": None, "sensitivity": None, "specificity": None, "loss_value": None, "threshold": None, "tn": None, "fp": None, "fn": None, "tp": None})
        else:
            item.update(
                {
                    "accuracy": metrics.get("accuracy"),
                    "f1": metrics.get("f1"),
                    "auc": metrics.get("auc"),
                    "sensitivity": metrics.get("sensitivity"),
                    "specificity": metrics.get("specificity"),
                    "loss_value": metrics.get("loss"),
                    "threshold": metrics.get("threshold"),
                    "tn": cm[0][0] if cm else None,
                    "fp": cm[0][1] if cm else None,
                    "fn": cm[1][0] if cm else None,
                    "tp": cm[1][1] if cm else None,
                }
            )
        summary.append(item)

    json_path = out_root / "summary_results.json"
    csv_path = out_root / "summary_results.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    fieldnames = ["model", "init", "train_data", "mri_used", "loss", "test", "accuracy", "f1", "auc", "sensitivity", "specificity", "tn", "fp", "fn", "tp", "loss_value", "threshold", "confusion_matrix", "metrics_file"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print("\n" + "=" * 120)
    print("SUMMARY RESULTS")
    print("=" * 120)
    columns = [
        ("Model", "model", 32),
        ("Init", "init", 16),
        ("Train data", "train_data", 14),
        ("MRI", "mri_used", 9),
        ("Loss", "loss", 12),
        ("Test", "test", 15),
        ("Acc", "accuracy", 7),
        ("F1", "f1", 7),
        ("AUC", "auc", 7),
        ("Sens", "sensitivity", 7),
        ("Spec", "specificity", 7),
        ("CM [[TN,FP],[FN,TP]]", "confusion_matrix", 22),
    ]
    header = " | ".join(name.ljust(width) for name, _, width in columns)
    print(header)
    print("-" * len(header))
    for row in summary:
        print(" | ".join(fmt(row[key])[:width].ljust(width) for _, key, width in columns))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    text_common = [
        "--max-len", str(args.max_len),
        "--wd", str(args.wd_text),
        "--warmup", str(args.warmup),
        "--threshold", str(args.threshold),
        "--seed", str(args.seed),
        "--workers", str(args.workers),
        "--accum", str(args.accum),
    ]
    paired_common = [
        "--images", args.images,
        "--epochs", str(args.paired_epochs),
        "--lr", str(args.paired_lr),
        "--batch", str(args.paired_batch),
        *text_common,
    ]
    paired_mri_common = [
        "--images", args.images,
        "--epochs", str(args.paired_epochs),
        "--lr", str(args.paired_lr),
        "--batch-text", str(args.paired_batch),
        "--batch-mri", str(args.mri_batch),
        *text_common,
    ]
    device_flags = []
    add_flag(device_flags, args.cpu, "--cpu")
    add_flag(device_flags, args.no_mgpu, "--no-mgpu")

    mri_dir = out_root / "00_mri_teacher"
    paired_only_dir = out_root / "01_paired_only_ce"
    large_text_dir = out_root / "02_large_text_ce"
    large_to_paired_ce_dir = out_root / "03_large_to_paired_ce"
    kd_dir = out_root / "04_large_to_paired_mri_kd"
    lupi_dir = out_root / "05_large_to_paired_lupi"
    kd_shuffle_dir = out_root / "06_large_to_paired_mri_kd_shuffled"
    lupi_shuffle_dir = out_root / "07_large_to_paired_lupi_shuffled"

    large_ckpt = large_text_dir / "best_auc_phobert"
    mri_ckpt = mri_dir / "best_auc_model.pt"

    stages = [
        (
            "0. MRI-only teacher: paired MRI train -> paired MRI test 280",
            [
                py, "train_mri.py",
                "--images", args.images,
                "--out", str(mri_dir),
                "--epochs", str(args.mri_epochs),
                "--lr", str(args.mri_lr),
                "--batch", str(args.mri_batch),
                "--wd", str(args.wd_mri),
                "--threshold", str(args.threshold),
                "--seed", str(args.seed),
                "--workers", str(args.workers),
                *device_flags,
            ],
        ),
        (
            "1. Paired-only CE: PhoBERT -> paired train -> paired test 280",
            [
                py, "train_pair_text.py",
                "--model", args.base_model,
                "--out", str(paired_only_dir),
                *paired_common,
                *device_flags,
            ],
        ),
        (
            "2. Large-text CE: PhoBERT -> ~20K train -> ~20K test",
            [
                py, "train_text.py",
                "--model", args.base_model,
                "--data", args.texts,
                "--out", str(large_text_dir),
                "--epochs", str(args.large_epochs),
                "--lr", str(args.large_lr),
                "--batch", str(args.large_batch),
                *text_common,
                *device_flags,
            ],
        ),
        (
            "3. Large-text -> paired CE",
            [
                py, "train_pair_text.py",
                "--model", str(large_ckpt),
                "--out", str(large_to_paired_ce_dir),
                *paired_common,
                *device_flags,
            ],
        ),
        (
            "4. Large-text -> paired MRI KD",
            [
                py, "kd_mri_text.py",
                "--student", str(large_ckpt),
                "--teacher", str(mri_ckpt),
                "--out", str(kd_dir),
                "--alpha", str(args.kd_alpha),
                "--kd", args.kd_loss,
                "--temp", str(args.kd_temp),
                *paired_mri_common,
                *device_flags,
            ],
        ),
        (
            "5. Large-text -> paired MRI LUPI",
            [
                py, "train_lupi.py",
                "--student", str(large_ckpt),
                "--teacher", str(mri_ckpt),
                "--out", str(lupi_dir),
                "--alpha-lupi", str(args.lupi_alpha),
                "--weight-min", str(args.lupi_weight_min),
                "--weight-max", str(args.lupi_weight_max),
                *paired_mri_common,
                *device_flags,
            ],
        ),
        (
            "6a. Shuffled KD control",
            [
                py, "kd_mri_text.py",
                "--student", str(large_ckpt),
                "--teacher", str(mri_ckpt),
                "--out", str(kd_shuffle_dir),
                "--alpha", str(args.kd_alpha),
                "--kd", args.kd_loss,
                "--temp", str(args.kd_temp),
                "--shuffle-teacher",
                *paired_mri_common,
                *device_flags,
            ],
        ),
        (
            "6b. Shuffled LUPI control",
            [
                py, "train_lupi.py",
                "--student", str(large_ckpt),
                "--teacher", str(mri_ckpt),
                "--out", str(lupi_shuffle_dir),
                "--alpha-lupi", str(args.lupi_alpha),
                "--weight-min", str(args.lupi_weight_min),
                "--weight-max", str(args.lupi_weight_max),
                "--shuffle-teacher",
                *paired_mri_common,
                *device_flags,
            ],
        ),
    ]

    manifest = {
        "images": args.images,
        "texts": args.texts,
        "out_root": str(out_root),
        "large_text_checkpoint": str(large_ckpt),
        "mri_teacher_checkpoint": str(mri_ckpt),
        "stages": [{"name": name, "command": cmd} for name, cmd in stages],
    }
    with open(out_root / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    for name, cmd in stages:
        run_stage(name, cmd, args.dry_run)

    summary_rows = [
        {
            "model": "MRI-only teacher",
            "init": "ImageNet",
            "train_data": "paired MRI",
            "mri_used": "Yes",
            "loss": "BCE",
            "test": "paired test 280",
            "dir": mri_dir,
            "split": "test",
        },
        {
            "model": "Paired-only CE",
            "init": "PhoBERT",
            "train_data": "paired text",
            "mri_used": "No",
            "loss": "CE",
            "test": "paired test 280",
            "dir": paired_only_dir,
            "split": "test",
        },
        {
            "model": "Large-text CE",
            "init": "PhoBERT",
            "train_data": "~20K text",
            "mri_used": "No",
            "loss": "CE",
            "test": "~20K test",
            "dir": large_text_dir,
            "split": "test",
        },
        {
            "model": "Large-text -> paired CE",
            "init": "Large-text ckpt",
            "train_data": "paired text",
            "mri_used": "No",
            "loss": "CE",
            "test": "paired test 280",
            "dir": large_to_paired_ce_dir,
            "split": "test",
        },
        {
            "model": "Large-text -> paired KD",
            "init": "Large-text ckpt",
            "train_data": "paired text",
            "mri_used": "Yes",
            "loss": "CE + KD",
            "test": "paired test 280",
            "dir": kd_dir,
            "split": "test",
        },
        {
            "model": "Large-text -> paired LUPI",
            "init": "Large-text ckpt",
            "train_data": "paired text",
            "mri_used": "Yes",
            "loss": "weighted CE",
            "test": "paired test 280",
            "dir": lupi_dir,
            "split": "test",
        },
        {
            "model": "Shuffled KD",
            "init": "Large-text ckpt",
            "train_data": "paired text",
            "mri_used": "shuffled",
            "loss": "CE + KD",
            "test": "paired test 280",
            "dir": kd_shuffle_dir,
            "split": "test",
        },
        {
            "model": "Shuffled LUPI",
            "init": "Large-text ckpt",
            "train_data": "paired text",
            "mri_used": "shuffled",
            "loss": "weighted CE",
            "test": "paired test 280",
            "dir": lupi_shuffle_dir,
            "split": "test",
        },
    ]
    if not args.dry_run:
        summarize_results(out_root, summary_rows)


if __name__ == "__main__":
    main()
