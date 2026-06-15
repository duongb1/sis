import argparse
import sys
import json
import numpy as np
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
        print_protocol_summary_report(output_dir, args.folds)


def print_protocol_summary_report(output_dir, folds):
    output_dir = Path(output_dir)
    
    def load_json(path):
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def get_binary_metrics(split_data):
        if not split_data:
            return None
        if isinstance(split_data, dict) and "binary_i63" in split_data:
            metrics = dict(split_data["binary_i63"])
        else:
            metrics = dict(split_data)
        
        # Ensure balanced accuracy is present
        if "balanced_accuracy" not in metrics and "sensitivity" in metrics and "specificity" in metrics:
            sens = metrics.get("sensitivity")
            spec = metrics.get("specificity")
            if sens is not None and spec is not None and not np.isnan(sens) and not np.isnan(spec):
                metrics["balanced_accuracy"] = float((sens + spec) / 2.0)
            else:
                metrics["balanced_accuracy"] = float("nan")
        return metrics

    def get_fold_stats(parent_folder, split_key):
        accuracy_list = []
        f1_list = []
        auc_list = []
        sens_list = []
        spec_list = []
        bal_acc_list = []
        
        for fold in range(folds):
            path = output_dir / parent_folder / f"fold_{fold}" / "metrics.json"
            data = load_json(path)
            if data:
                metrics = get_binary_metrics(data.get(split_key))
                if metrics:
                    if metrics.get("accuracy") is not None:
                        accuracy_list.append(metrics.get("accuracy"))
                    if metrics.get("f1") is not None:
                        f1_list.append(metrics.get("f1"))
                    if metrics.get("auc") is not None:
                        auc_list.append(metrics.get("auc"))
                    if metrics.get("sensitivity") is not None:
                        sens_list.append(metrics.get("sensitivity"))
                    if metrics.get("specificity") is not None:
                        spec_list.append(metrics.get("specificity"))
                    if metrics.get("balanced_accuracy") is not None:
                        bal_acc_list.append(metrics.get("balanced_accuracy"))
        
        def mean_std(lst):
            vals = [v for v in lst if v is not None and not np.isnan(v)]
            if not vals:
                return None
            return np.mean(vals), np.std(vals)
            
        return {
            "accuracy": mean_std(accuracy_list),
            "f1": mean_std(f1_list),
            "auc": mean_std(auc_list),
            "sensitivity": mean_std(sens_list),
            "specificity": mean_std(spec_list),
            "balanced_accuracy": mean_std(bal_acc_list),
        }

    # Load Model 2
    model2_path = output_dir / "model_2_processed_9937_random" / "metrics.json"
    model2_data = load_json(model2_path)
    model2_test = get_binary_metrics(model2_data.get("test")) if model2_data else None
    model2_cross = get_binary_metrics(model2_data.get("processed_700_all_eval")) if model2_data else None

    # Load Model 1
    model1_stats = get_fold_stats("model_1_processed_700_5fold", "test")
    model1_cross_val = get_fold_stats("model_1_processed_700_5fold", "processed_9937_random_val")
    model1_cross_test = get_fold_stats("model_1_processed_700_5fold", "processed_9937_random_test")

    # Load Model 3
    model3_stats = get_fold_stats("model_3_large_to_small_finetune_5fold", "test")
    model3_cross_val = get_fold_stats("model_3_large_to_small_finetune_5fold", "processed_9937_random_val")
    model3_cross_test = get_fold_stats("model_3_large_to_small_finetune_5fold", "processed_9937_random_test")

    print("\n" + "=" * 80)
    print("                      FINAL PROTOCOL SUMMARY REPORT")
    print("=" * 80)

    # PART 1
    print("\nPART 1: Model 1 (Processed 700 - 5-Fold)")
    print("-" * 80)
    print("Reported on Held-Out Test Fold (mean ± std over 5 folds):")
    for metric_name, key in [
        ("Accuracy", "accuracy"),
        ("F1-score", "f1"),
        ("AUC", "auc"),
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
        ("Balanced Accuracy", "balanced_accuracy"),
    ]:
        val = model1_stats.get(key)
        if val and val[0] is not None:
            print(f"- {metric_name:<19}: {val[0]*100:>6.2f}% ± {val[1]*100:.2f}%")
        else:
            print(f"- {metric_name:<19}: N/A")

    # PART 2
    print("\nPART 2: Model 2 (Processed 9937 - Random Split)")
    print("-" * 80)
    print("Reported on Test Split (single run):")
    if model2_test:
        for metric_name, key in [
            ("Accuracy", "accuracy"),
            ("F1-score", "f1"),
            ("AUC", "auc"),
            ("Sensitivity", "sensitivity"),
            ("Specificity", "specificity"),
            ("Balanced Accuracy", "balanced_accuracy"),
        ]:
            val = model2_test.get(key)
            if val is not None and not np.isnan(val):
                print(f"- {metric_name:<19}: {val*100:>6.2f}%")
            else:
                print(f"- {metric_name:<19}: N/A")
    else:
        print("Metrics file not found or empty.")

    # PART 3
    print("\nPART 3: Cross Evaluation")
    print("-" * 80)
    print("A) Model 2 evaluated on ALL Processed 700 (single run):")
    if model2_cross:
        for metric_name, key in [
            ("Accuracy", "accuracy"),
            ("F1-score", "f1"),
            ("AUC", "auc"),
            ("Sensitivity", "sensitivity"),
            ("Specificity", "specificity"),
            ("Balanced Accuracy", "balanced_accuracy"),
        ]:
            val = model2_cross.get(key)
            if val is not None and not np.isnan(val):
                print(f"- {metric_name:<19}: {val*100:>6.2f}%")
            else:
                print(f"- {metric_name:<19}: N/A")
    else:
        print("Metrics file not found or empty.")

    print("\nB) Model 1 folds evaluated on Model 2 splits (mean ± std over 5 folds):")
    print("* On Model 2 Validation Split:")
    for metric_name, key in [
        ("Accuracy", "accuracy"),
        ("F1-score", "f1"),
        ("AUC", "auc"),
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
        ("Balanced Accuracy", "balanced_accuracy"),
    ]:
        val = model1_cross_val.get(key)
        if val and val[0] is not None:
            print(f"  - {metric_name:<17}: {val[0]*100:>6.2f}% ± {val[1]*100:.2f}%")
        else:
            print(f"  - {metric_name:<17}: N/A")

    print("* On Model 2 Test Split:")
    for metric_name, key in [
        ("Accuracy", "accuracy"),
        ("F1-score", "f1"),
        ("AUC", "auc"),
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
        ("Balanced Accuracy", "balanced_accuracy"),
    ]:
        val = model1_cross_test.get(key)
        if val and val[0] is not None:
            print(f"  - {metric_name:<17}: {val[0]*100:>6.2f}% ± {val[1]*100:.2f}%")
        else:
            print(f"  - {metric_name:<17}: N/A")

    # PART 4
    print("\nPART 4: Model 3 (Fine-tuned Model 2 Checkpoint on Processed 700 Folds)")
    print("-" * 80)
    print("Reported on Held-Out Test Fold (mean ± std over 5 folds):")
    for metric_name, key in [
        ("Accuracy", "accuracy"),
        ("F1-score", "f1"),
        ("AUC", "auc"),
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
        ("Balanced Accuracy", "balanced_accuracy"),
    ]:
        val = model3_stats.get(key)
        if val and val[0] is not None:
            print(f"- {metric_name:<19}: {val[0]*100:>6.2f}% ± {val[1]*100:.2f}%")
        else:
            print(f"- {metric_name:<19}: N/A")

    print("\nReported on Model 2 Validation Split (mean ± std over 5 folds):")
    for metric_name, key in [
        ("Accuracy", "accuracy"),
        ("F1-score", "f1"),
        ("AUC", "auc"),
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
        ("Balanced Accuracy", "balanced_accuracy"),
    ]:
        val = model3_cross_val.get(key)
        if val and val[0] is not None:
            print(f"- {metric_name:<19}: {val[0]*100:>6.2f}% ± {val[1]*100:.2f}%")
        else:
            print(f"- {metric_name:<19}: N/A")

    print("\nReported on Model 2 Test Split (mean ± std over 5 folds):")
    for metric_name, key in [
        ("Accuracy", "accuracy"),
        ("F1-score", "f1"),
        ("AUC", "auc"),
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
        ("Balanced Accuracy", "balanced_accuracy"),
    ]:
        val = model3_cross_test.get(key)
        if val and val[0] is not None:
            print(f"- {metric_name:<19}: {val[0]*100:>6.2f}% ± {val[1]*100:.2f}%")
        else:
            print(f"- {metric_name:<19}: N/A")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
