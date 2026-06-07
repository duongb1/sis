import argparse
import sys
from pathlib import Path

from sislib.reports import aggregate_experiment, print_final_small_report
from sislib.runner_utils import run_stage


ROOT = Path(__file__).resolve().parent
DEFAULT_EXCEL_ROOT = "/kaggle/input/datasets/duongbui/siscth"


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
        "name": "large_multitask",
        "data": "9937_co_label.xlsx,9937_khong_label.xlsx",
        "task": "multitask",
        "positive": "I63_INFARCTION",
    },
    {
        "name": "small_multiclass",
        "data": "700_co_label.xlsx,700_khong_label.xlsx",
        "task": "multiclass",
        "positive": "I63_INFARCTION",
    },
    {
        "name": "small_multitask",
        "data": "700_co_label.xlsx,700_khong_label.xlsx",
        "task": "multitask",
        "positive": "I63_INFARCTION",
    },
]

EXPERIMENT_ALIASES = {
    "large": {"large_binary", "large_multiclass", "large_multitask"},
    "small": {"small_binary", "small_multiclass", "small_multitask"},
}


def parse_args():
    p = argparse.ArgumentParser(description="Run Excel SIS text training for large/small binary and multi-class with 5-fold 70/10/20 splits.")
    p.add_argument("--excel-root", default=DEFAULT_EXCEL_ROOT, help="Folder containing the four Excel files.")
    p.add_argument("--output-dir", default="/kaggle/working/sis_excel_5fold_fieldaware_binary_mcstrat")
    p.add_argument("--model", default="vinai/phobert-base")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50", help="Threshold sweep written to metrics.json for binary_i63.")
    p.add_argument("--pooling", choices=["cls", "attention", "gated"], default="attention", help="Pooling method after PhoBERT encoder.")
    p.add_argument("--input-mode", choices=["concat", "field"], default="field", help="Input representation mode: concat all fields or encode Excel fields separately.")
    p.add_argument("--max-len-per-field", type=int, default=128)
    p.add_argument("--save-field-attention", action="store_true", help="Save field-level attention weights in field-aware prediction CSVs.")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.2, help="Documented protocol ratio. With 5 folds, test is one fold = 0.2.")
    p.add_argument("--excel-split-label", choices=["target", "binary", "multiclass"], default="multiclass", help="Label source used only to stratify Excel kfold splits.")
    p.add_argument("--only", default="small_binary", help="Comma-separated experiment names to run. Use --only large for all large experiments, --only small for all small experiments, or --only all for every experiment.")
    p.add_argument("--lambda-aux", "--aux-weight", dest="lambda_aux", type=float, default=0.5, help="Auxiliary 3-class loss weight for multitask experiments.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    return p.parse_args()


def selected_experiments(value):
    if not value or value.strip().lower() == "all":
        return EXPERIMENTS
    requested = {item.strip() for item in value.split(",") if item.strip()}
    names = set()
    for item in requested:
        names.update(EXPERIMENT_ALIASES.get(item.lower(), {item}))
    unknown = names - {experiment["name"] for experiment in EXPERIMENTS}
    if unknown:
        aliases = ", ".join(sorted(EXPERIMENT_ALIASES))
        raise ValueError(f"Unknown experiment names: {', '.join(sorted(unknown))}. Available aliases: all, {aliases}")
    return [experiment for experiment in EXPERIMENTS if experiment["name"] in names]


def train_cmd(args, experiment, fold, out):
    excel_root = args.excel_root.rstrip("/\\")
    data = ",".join(f"{excel_root}/{name}" for name in experiment["data"].split(","))
    input_mode = "concat" if experiment["task"] == "multitask" and args.input_mode == "field" else args.input_mode
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
        "excel",
        "--excel-task",
        experiment["task"],
        "--split-strategy",
        "kfold",
        "--n-folds",
        args.folds,
        "--fold-index",
        fold,
        "--excel-split-label",
        args.excel_split_label,
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
        "--thresholds",
        args.thresholds,
        "--pooling",
        args.pooling,
        "--input-mode",
        input_mode,
        "--max-len-per-field",
        args.max_len_per_field,
        "--workers",
        args.workers,
    ]
    if args.save_field_attention:
        cmd.append("--save-field-attention")
    if experiment["task"] == "multitask":
        cmd.extend(["--lambda-aux", args.lambda_aux])
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
    print(f"Excel kfold stratify label: {args.excel_split_label}")
    if args.input_mode == "field" and any(experiment["task"] == "multitask" for experiment in selected_experiments(args.only)):
        print("Note: multitask experiments use --input-mode concat because field mode is binary/multiclass only.")

    experiments = selected_experiments(args.only)
    for experiment in experiments:
        for fold in range(args.folds):
            out = f"{args.output_dir.rstrip('/\\')}/{experiment['name']}/fold_{fold}"
            run_stage(
                f"{experiment['name']} fold {fold}: train=70%, val=10%, test=20%",
                train_cmd(args, experiment, fold, out),
                Path(out) / "metrics.json",
                force=args.force,
                dry_run=args.dry_run,
                cwd=ROOT,
            )
        if not args.dry_run:
            aggregate_experiment(output_dir, experiment["name"], args.folds)

    if not args.dry_run:
        print_final_small_report(output_dir, args)


if __name__ == "__main__":
    main()
