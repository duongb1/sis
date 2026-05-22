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
    p.add_argument("--class-weight-co", type=float, default=1.2)
    p.add_argument("--hn-text-threshold", type=float, default=0.7)
    p.add_argument("--hn-mri-negative-threshold", type=float, default=0.3)
    p.add_argument("--hn-mri-ambiguous-threshold", type=float, default=0.5)
    p.add_argument("--hn-hard-weight", type=float, default=3.0)
    p.add_argument("--hn-ambiguous-weight", type=float, default=0.5)
    p.add_argument("--hn-epochs", type=int, default=5)
    p.add_argument("--hn-lr", type=float, default=1e-5)
    p.add_argument("--lupi-alpha", type=float, default=0.2)
    p.add_argument("--lupi-weight-min", type=float, default=0.75)
    p.add_argument("--lupi-weight-max", type=float, default=1.25)
    p.add_argument("--lambda-align", type=float, default=0.05)
    p.add_argument("--align-loss", choices=["cosine", "mse"], default="cosine")
    p.add_argument("--aux-dim", type=int, default=256)
    p.add_argument("--align-warmup-epochs", type=int, default=3)
    p.add_argument("--detach-aux", action="store_true")

    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    p.add_argument("--include-control", action="store_true")
    p.add_argument("--include-ablation", action="store_true")
    p.add_argument("--include-all", action="store_true")
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


def build_summary(rows):
    summary = []
    for row in rows:
        metrics = read_split_metrics(row["dir"], row["split"])
        cm = read_confusion_matrix(row["dir"], row["split"])
        item = {k: row[k] for k in ["group", "model", "init", "train_data", "mri_used", "loss", "test"]}
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
    return summary


def print_summary_table(title, summary):
    print("\n" + "=" * 120)
    print(title)
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


def summarize_results(out_root, rows):
    summary = build_summary(rows)

    json_path = out_root / "summary_results.json"
    csv_path = out_root / "summary_results.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    fieldnames = ["group", "model", "init", "train_data", "mri_used", "loss", "test", "accuracy", "f1", "auc", "sensitivity", "specificity", "tn", "fp", "fn", "tp", "loss_value", "threshold", "confusion_matrix", "metrics_file"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print_summary_table("MAIN RESULTS", [row for row in summary if row["group"] == "main"])
    print_summary_table("CONTROL EXPERIMENTS", [row for row in summary if row["group"] == "control"])
    print_summary_table("ABLATION / APPENDIX", [row for row in summary if row["group"] == "ablation"])
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    if args.dry_run:
        try:
            out_root.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    else:
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
    large_direct_dir = out_root / "03_large_text_direct_paired"
    large_to_paired_ce_dir = out_root / "04_large_to_paired_ce"
    hn_dir = out_root / "05_mri_hard_negative_reweight"
    hn_shuffle_dir = out_root / "06_mri_hard_negative_reweight_shuffled"

    paired_kd_dir = out_root / "90_paired_mri_kd"
    paired_lupi_dir = out_root / "91_paired_lupi"
    paired_dual_dir = out_root / "92_paired_dual_mri_align"
    weighted_ce_dir = out_root / "93_large_to_paired_weighted_ce"
    kd_dir = out_root / "94_large_to_paired_mri_kd"
    conf_kd_dir = out_root / "95_large_to_paired_mri_kd_conf"
    lupi_dir = out_root / "96_large_to_paired_lupi"
    large_dual_dir = out_root / "97_large_to_paired_dual_mri_align"

    large_ckpt = large_text_dir / "best_auc_phobert"
    mri_ckpt = mri_dir / "best_auc_model.pt"

    main_stages = [
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
            "3. Large-text direct on paired: no paired fine-tuning",
            [
                py, "eval_pair_text.py",
                "--model", str(large_ckpt),
                "--out", str(large_direct_dir),
                "--images", args.images,
                "--splits", "train", "val", "test",
                "--max-len", str(args.max_len),
                "--batch", str(args.paired_batch),
                "--threshold", str(args.threshold),
                "--seed", str(args.seed),
                "--workers", str(args.workers),
                *device_flags,
            ],
        ),
        (
            "4. Large-text -> paired CE",
            [
                py, "train_pair_text.py",
                "--model", str(large_ckpt),
                "--out", str(large_to_paired_ce_dir),
                *paired_common,
                *device_flags,
            ],
        ),
        (
            "5. MRI-guided hard-negative reweighting",
            [
                py, "train_hard_negative_reweight.py",
                "--student", str(large_ckpt),
                "--teacher", str(mri_ckpt),
                "--out", str(hn_dir),
                "--images", args.images,
                "--epochs", str(args.hn_epochs),
                "--lr", str(args.hn_lr),
                "--batch-text", str(args.paired_batch),
                "--batch-mri", str(args.mri_batch),
                "--max-len", str(args.max_len),
                "--wd", str(args.wd_text),
                "--warmup", str(args.warmup),
                "--threshold", str(args.threshold),
                "--seed", str(args.seed),
                "--workers", str(args.workers),
                "--accum", str(args.accum),
                "--text-fp-threshold", str(args.hn_text_threshold),
                "--mri-negative-threshold", str(args.hn_mri_negative_threshold),
                "--mri-ambiguous-threshold", str(args.hn_mri_ambiguous_threshold),
                "--hard-negative-weight", str(args.hn_hard_weight),
                "--ambiguous-negative-weight", str(args.hn_ambiguous_weight),
                *device_flags,
            ],
        ),
    ]

    ablation_stages = [
        (
            "A1. Paired-only MRI KD",
            [
                py, "kd_mri_text.py",
                "--student", args.base_model,
                "--teacher", str(mri_ckpt),
                "--out", str(paired_kd_dir),
                "--alpha", str(args.kd_alpha),
                "--kd", args.kd_loss,
                "--temp", str(args.kd_temp),
                *paired_mri_common,
                *device_flags,
            ],
        ),
        (
            "A2. Paired-only MRI LUPI",
            [
                py, "train_lupi.py",
                "--student", args.base_model,
                "--teacher", str(mri_ckpt),
                "--out", str(paired_lupi_dir),
                "--alpha-lupi", str(args.lupi_alpha),
                "--weight-min", str(args.lupi_weight_min),
                "--weight-max", str(args.lupi_weight_max),
                *paired_mri_common,
                *device_flags,
            ],
        ),
        (
            "A3. Paired-only Dual MRI-Align",
            [
                py, "train_dual_mri_align.py",
                "--student", args.base_model,
                "--teacher", str(mri_ckpt),
                "--out", str(paired_dual_dir),
                "--lambda-align", str(args.lambda_align),
                "--align-loss", args.align_loss,
                "--aux-dim", str(args.aux_dim),
                "--align-warmup-epochs", str(args.align_warmup_epochs),
                *paired_mri_common,
                *device_flags,
            ] + (["--detach-aux"] if args.detach_aux else []),
        ),
        (
            "A4. Large-text -> paired class-weighted CE",
            [
                py, "train_pair_text.py",
                "--model", str(large_ckpt),
                "--out", str(weighted_ce_dir),
                "--class-weight-co", str(args.class_weight_co),
                *paired_common,
                *device_flags,
            ],
        ),
        (
            "A5. Large-text -> paired MRI KD",
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
            "A6. Large-text -> paired confidence-aware MRI KD",
            [
                py, "kd_mri_text.py",
                "--student", str(large_ckpt),
                "--teacher", str(mri_ckpt),
                "--out", str(conf_kd_dir),
                "--alpha", str(args.kd_alpha),
                "--kd", args.kd_loss,
                "--kd-weight", "confidence",
                "--temp", str(args.kd_temp),
                *paired_mri_common,
                *device_flags,
            ],
        ),
        (
            "A7. Large-text -> paired MRI LUPI",
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
            "A8. Large-text -> paired Dual MRI-Align",
            [
                py, "train_dual_mri_align.py",
                "--student", str(large_ckpt),
                "--teacher", str(mri_ckpt),
                "--out", str(large_dual_dir),
                "--lambda-align", str(args.lambda_align),
                "--align-loss", args.align_loss,
                "--aux-dim", str(args.aux_dim),
                "--align-warmup-epochs", str(args.align_warmup_epochs),
                *paired_mri_common,
                *device_flags,
            ] + (["--detach-aux"] if args.detach_aux else []),
        ),
    ]

    control_stages = [
        (
            "C1. Shuffled MRI-guided hard-negative reweighting",
            [
                py, "train_hard_negative_reweight.py",
                "--student", str(large_ckpt),
                "--teacher", str(mri_ckpt),
                "--out", str(hn_shuffle_dir),
                "--images", args.images,
                "--epochs", str(args.hn_epochs),
                "--lr", str(args.hn_lr),
                "--batch-text", str(args.paired_batch),
                "--batch-mri", str(args.mri_batch),
                "--max-len", str(args.max_len),
                "--wd", str(args.wd_text),
                "--warmup", str(args.warmup),
                "--threshold", str(args.threshold),
                "--seed", str(args.seed),
                "--workers", str(args.workers),
                "--accum", str(args.accum),
                "--text-fp-threshold", str(args.hn_text_threshold),
                "--mri-negative-threshold", str(args.hn_mri_negative_threshold),
                "--mri-ambiguous-threshold", str(args.hn_mri_ambiguous_threshold),
                "--hard-negative-weight", str(args.hn_hard_weight),
                "--ambiguous-negative-weight", str(args.hn_ambiguous_weight),
                "--shuffle-teacher",
                *device_flags,
            ],
        ),
    ]
    stages = list(main_stages)
    if args.include_ablation or args.include_all:
        stages.extend(ablation_stages)
    if args.include_control or args.include_all:
        stages.extend(control_stages)

    manifest = {
        "images": args.images,
        "texts": args.texts,
        "out_root": str(out_root),
        "large_text_checkpoint": str(large_ckpt),
        "mri_teacher_checkpoint": str(mri_ckpt),
        "include_control": bool(args.include_control or args.include_all),
        "include_ablation": bool(args.include_ablation or args.include_all),
        "stages": [{"name": name, "command": cmd} for name, cmd in stages],
    }
    if out_root.exists():
        with open(out_root / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    for name, cmd in stages:
        run_stage(name, cmd, args.dry_run)

    summary_rows = [
        {
            "group": "main",
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
            "group": "main",
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
            "group": "main",
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
            "group": "main",
            "model": "Large-text direct",
            "init": "Large-text ckpt",
            "train_data": "none",
            "mri_used": "No",
            "loss": "eval only",
            "test": "paired test 280",
            "dir": large_direct_dir,
            "split": "test",
        },
        {
            "group": "main",
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
            "group": "main",
            "model": "MRI hard-neg reweight",
            "init": "Large-text ckpt",
            "train_data": "paired text",
            "mri_used": "Yes",
            "loss": "weighted CE",
            "test": "paired test 280",
            "dir": hn_dir,
            "split": "test",
        },
    ]
    if args.include_ablation or args.include_all:
        summary_rows.extend([
        {
            "group": "ablation",
            "model": "Paired-only MRI-KD",
            "init": "PhoBERT",
            "train_data": "paired text",
            "mri_used": "Yes",
            "loss": "CE + KD",
            "test": "paired test 280",
            "dir": paired_kd_dir,
            "split": "test",
        },
        {
            "group": "ablation",
            "model": "Paired-only MRI-LUPI",
            "init": "PhoBERT",
            "train_data": "paired text",
            "mri_used": "Yes",
            "loss": "weighted CE",
            "test": "paired test 280",
            "dir": paired_lupi_dir,
            "split": "test",
        },
        {
            "group": "ablation",
            "model": "Paired-only Dual MRI-Align",
            "init": "PhoBERT",
            "train_data": "paired text",
            "mri_used": "Yes",
            "loss": "CE + align",
            "test": "paired test 280",
            "dir": paired_dual_dir,
            "split": "test",
        },
        {
            "group": "ablation",
            "model": "Large-text -> weighted CE",
            "init": "Large-text ckpt",
            "train_data": "paired text",
            "mri_used": "No",
            "loss": "weighted CE",
            "test": "paired test 280",
            "dir": weighted_ce_dir,
            "split": "test",
        },
        {
            "group": "ablation",
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
            "group": "ablation",
            "model": "Large-text -> conf KD",
            "init": "Large-text ckpt",
            "train_data": "paired text",
            "mri_used": "Yes",
            "loss": "conf CE+KD",
            "test": "paired test 280",
            "dir": conf_kd_dir,
            "split": "test",
        },
        {
            "group": "ablation",
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
            "group": "ablation",
            "model": "Large-text -> Dual MRI-Align",
            "init": "Large-text ckpt",
            "train_data": "paired text",
            "mri_used": "Yes",
            "loss": "CE + align",
            "test": "paired test 280",
            "dir": large_dual_dir,
            "split": "test",
        },
        ])
    if args.include_control or args.include_all:
        summary_rows.extend([
        {
            "group": "control",
            "model": "Shuffled hard-neg reweight",
            "init": "Large-text ckpt",
            "train_data": "paired text",
            "mri_used": "shuffled",
            "loss": "weighted CE",
            "test": "paired test 280",
            "dir": hn_shuffle_dir,
            "split": "test",
        },
        ])
    if not args.dry_run:
        summarize_results(out_root, summary_rows)


if __name__ == "__main__":
    main()
