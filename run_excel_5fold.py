import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
DEFAULT_EXCEL_ROOT = "/kaggle/input/datasets/duongb/cthsis"


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
    p.add_argument("--excel-root", default=DEFAULT_EXCEL_ROOT, help="Folder containing the four Excel files.")
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
    p.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50", help="Threshold sweep written to metrics.json for binary_i63.")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.2, help="Documented protocol ratio. With 5 folds, test is one fold = 0.2.")
    p.add_argument("--only", default="small_binary,small_multiclass", help="Comma-separated experiment names to run. Default runs only small_binary and small_multiclass. Use --only all to run every experiment.")
    p.add_argument("--no-ensemble", action="store_true", help="Do not run small_binary + small_multiclass score-level ensemble after folds finish.")
    p.add_argument("--ensemble-mode", choices=["fixed", "tuned"], default="fixed", help="fixed uses one beta/threshold for every fold; tuned selects beta/threshold on validation.")
    p.add_argument("--ensemble-beta", type=float, default=0.5, help="Fixed ensemble beta for --ensemble-mode fixed.")
    p.add_argument("--ensemble-threshold", type=float, default=0.5, help="Fixed ensemble threshold for --ensemble-mode fixed.")
    p.add_argument("--ensemble-betas", default=None, help="Comma-separated beta grid. Defaults to 0.00,0.05,...,1.00.")
    p.add_argument("--ensemble-thresholds", default=None, help="Comma-separated threshold grid. Defaults to 0.20,0.21,...,0.80.")
    p.add_argument("--ensemble-min-sensitivity", type=float, default=0.83)
    p.add_argument("--ensemble-objective", choices=["max_spec_with_sens_constraint", "max_f1", "max_balanced_accuracy"], default="max_spec_with_sens_constraint")
    p.add_argument("--no-risk-score", action="store_true", help="Do not run small_multiclass I63-oriented risk score after folds finish.")
    p.add_argument("--risk-alphas", default="0.00,0.05,0.10,0.15,0.20,0.30")
    p.add_argument("--risk-thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55")
    p.add_argument("--risk-min-sensitivity", type=float, default=0.80)
    p.add_argument("--risk-objective", choices=["max_spec_with_sens_constraint", "max_f1", "max_balanced_accuracy"], default="max_spec_with_sens_constraint")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    return p.parse_args()


def selected_experiments(value):
    if not value or value.strip().lower() == "all":
        return EXPERIMENTS
    names = {item.strip() for item in value.split(",") if item.strip()}
    unknown = names - {experiment["name"] for experiment in EXPERIMENTS}
    if unknown:
        raise ValueError(f"Unknown experiment names: {', '.join(sorted(unknown))}")
    return [experiment for experiment in EXPERIMENTS if experiment["name"] in names]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_numeric(prefix, value, out):
    if isinstance(value, dict):
        for key, child in value.items():
            flatten_numeric(f"{prefix}.{key}" if prefix else key, child, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)


def summarize_metric_rows(rows):
    keys = sorted(set().union(*(row.keys() for row in rows)))
    summary = {}
    for key in keys:
        values = np.array([row[key] for row in rows if key in row and not np.isnan(row[key])], dtype=np.float64)
        if values.size == 0:
            continue
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "n": int(values.size),
        }
    return summary


def write_summary_csv(path, summary):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "n", "mean_plus_minus_std"])
        writer.writeheader()
        for metric, stats in summary.items():
            writer.writerow(
                {
                    "metric": metric,
                    "mean": f"{stats['mean']:.6f}",
                    "std": f"{stats['std']:.6f}",
                    "n": stats["n"],
                    "mean_plus_minus_std": f"{stats['mean']:.3f} ± {stats['std']:.3f}",
                }
            )


def binary_metrics(y_true, score, threshold):
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    pred = (score >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y_true, score))
    except ValueError:
        auc = float("nan")
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def metric_key(metrics, objective):
    if objective == "max_spec_with_sens_constraint":
        return (metrics["specificity"], metrics["f1"], metrics["balanced_accuracy"])
    if objective == "max_f1":
        return (metrics["f1"], metrics["balanced_accuracy"], metrics["specificity"])
    if objective == "max_balanced_accuracy":
        return (metrics["balanced_accuracy"], metrics["f1"], metrics["specificity"])
    raise ValueError(f"Unknown objective: {objective}")


def tune_ensemble(p_binary, p_multi, y_true, betas, thresholds, min_sensitivity, objective):
    best = None
    current_min_sensitivity = min_sensitivity
    while best is None and current_min_sensitivity >= 0:
        for beta in betas:
            score = beta * p_binary + (1.0 - beta) * p_multi
            for threshold in thresholds:
                metrics = binary_metrics(y_true, score, threshold)
                if objective == "max_spec_with_sens_constraint" and metrics["sensitivity"] < current_min_sensitivity:
                    continue
                candidate = {
                    "beta": float(beta),
                    "threshold": float(threshold),
                    "min_sensitivity": float(current_min_sensitivity),
                    "metrics": metrics,
                    "key": metric_key(metrics, objective),
                }
                if best is None or candidate["key"] > best["key"]:
                    best = candidate
        current_min_sensitivity -= 0.01
    if best is None:
        raise RuntimeError("Could not tune ensemble.")
    best.pop("key", None)
    return best


def tune_risk_score(p_i63, p_other_cbv, y_true, alphas, thresholds, min_sensitivity, objective):
    best = None
    current_min_sensitivity = min_sensitivity
    while best is None and current_min_sensitivity >= 0:
        for alpha in alphas:
            score = p_i63 + alpha * p_other_cbv
            for threshold in thresholds:
                metrics = binary_metrics(y_true, score, threshold)
                if objective == "max_spec_with_sens_constraint" and metrics["sensitivity"] < current_min_sensitivity:
                    continue
                candidate = {
                    "alpha": float(alpha),
                    "threshold": float(threshold),
                    "min_sensitivity": float(current_min_sensitivity),
                    "metrics": metrics,
                    "key": metric_key(metrics, objective),
                }
                if best is None or candidate["key"] > best["key"]:
                    best = candidate
        current_min_sensitivity -= 0.01
    if best is None:
        raise RuntimeError("Could not tune risk score.")
    best.pop("key", None)
    return best


def read_binary_predictions(path):
    rows = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item_id = row.get("id")
            if not item_id:
                continue
            rows[item_id] = {
                "y": int(row["true_binary_i63"]),
                "score": float(row["prob_binary_i63"]),
            }
    return rows


def read_risk_predictions(path):
    rows = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item_id = row.get("id")
            if not item_id:
                continue
            rows[item_id] = {
                "y": int(row["true_binary_i63"]),
                "p_i63": float(row["prob_I63_INFARCTION"]),
                "p_other_cbv": float(row["prob_OTHER_CEREBROVASCULAR"]),
            }
    return rows


def align_risk_predictions(rows):
    ids = sorted(rows)
    y = np.array([rows[item_id]["y"] for item_id in ids], dtype=np.int64)
    p_i63 = np.array([rows[item_id]["p_i63"] for item_id in ids], dtype=np.float64)
    p_other_cbv = np.array([rows[item_id]["p_other_cbv"] for item_id in ids], dtype=np.float64)
    return ids, y, p_i63, p_other_cbv


def align_predictions(binary_rows, multi_rows):
    common_ids = sorted(set(binary_rows) & set(multi_rows))
    if len(common_ids) != len(binary_rows) or len(common_ids) != len(multi_rows):
        raise RuntimeError(
            f"Prediction IDs do not align: binary={len(binary_rows)} multi={len(multi_rows)} common={len(common_ids)}"
        )
    y = np.array([binary_rows[item_id]["y"] for item_id in common_ids], dtype=np.int64)
    y_multi = np.array([multi_rows[item_id]["y"] for item_id in common_ids], dtype=np.int64)
    if not np.array_equal(y, y_multi):
        raise RuntimeError("Binary and multiclass predictions disagree on true binary labels.")
    p_binary = np.array([binary_rows[item_id]["score"] for item_id in common_ids], dtype=np.float64)
    p_multi = np.array([multi_rows[item_id]["score"] for item_id in common_ids], dtype=np.float64)
    return common_ids, y, p_binary, p_multi


def parse_float_grid(value, default):
    if value is None:
        return np.array(default, dtype=np.float64)
    return np.array([float(item.strip()) for item in str(value).split(",") if item.strip()], dtype=np.float64)


def write_ensemble_predictions(path, ids, y_true, p_binary, p_multi, beta, threshold):
    score = beta * p_binary + (1.0 - beta) * p_multi
    pred = (score >= threshold).astype(np.int64)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "true_binary_i63", "pred_binary_i63", "score_ensemble", "prob_binary_model", "prob_multiclass_model", "beta", "threshold"],
        )
        writer.writeheader()
        for item_id, y, prediction, ens_score, binary_score, multi_score in zip(ids, y_true, pred, score, p_binary, p_multi):
            writer.writerow(
                {
                    "id": item_id,
                    "true_binary_i63": int(y),
                    "pred_binary_i63": int(prediction),
                    "score_ensemble": f"{ens_score:.6f}",
                    "prob_binary_model": f"{binary_score:.6f}",
                    "prob_multiclass_model": f"{multi_score:.6f}",
                    "beta": f"{beta:.6f}",
                    "threshold": f"{threshold:.6f}",
                }
            )


def write_risk_predictions(path, ids, y_true, p_i63, p_other_cbv, alpha, threshold):
    score = p_i63 + alpha * p_other_cbv
    pred = (score >= threshold).astype(np.int64)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "true_binary_i63",
                "pred_binary_i63",
                "risk_score",
                "prob_I63_INFARCTION",
                "prob_OTHER_CEREBROVASCULAR",
                "alpha",
                "threshold",
            ],
        )
        writer.writeheader()
        for item_id, y, prediction, risk_score, i63_score, other_score in zip(ids, y_true, pred, score, p_i63, p_other_cbv):
            writer.writerow(
                {
                    "id": item_id,
                    "true_binary_i63": int(y),
                    "pred_binary_i63": int(prediction),
                    "risk_score": f"{risk_score:.6f}",
                    "prob_I63_INFARCTION": f"{i63_score:.6f}",
                    "prob_OTHER_CEREBROVASCULAR": f"{other_score:.6f}",
                    "alpha": f"{alpha:.6f}",
                    "threshold": f"{threshold:.6f}",
                }
            )


def run_small_ensemble(output_dir, folds, mode, fixed_beta, fixed_threshold, betas, thresholds, min_sensitivity, objective):
    ensemble_dir = output_dir / "small_ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    summary_rows = []

    for fold in range(folds):
        binary_dir = output_dir / "small_binary" / f"fold_{fold}"
        multi_dir = output_dir / "small_multiclass" / f"fold_{fold}"
        required = [
            binary_dir / "val_predictions_best_auc.csv",
            multi_dir / "val_predictions_best_auc.csv",
            binary_dir / "test_predictions_best_auc.csv",
            multi_dir / "test_predictions_best_auc.csv",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            print(f"Skip small ensemble: missing prediction files for fold {fold}.")
            return None

        val_ids, y_val, p_bin_val, p_multi_val = align_predictions(
            read_binary_predictions(binary_dir / "val_predictions_best_auc.csv"),
            read_binary_predictions(multi_dir / "val_predictions_best_auc.csv"),
        )
        test_ids, y_test, p_bin_test, p_multi_test = align_predictions(
            read_binary_predictions(binary_dir / "test_predictions_best_auc.csv"),
            read_binary_predictions(multi_dir / "test_predictions_best_auc.csv"),
        )
        if mode == "fixed":
            beta = fixed_beta
            threshold = fixed_threshold
            val_score = beta * p_bin_val + (1.0 - beta) * p_multi_val
            best = {
                "mode": "fixed",
                "beta": float(beta),
                "threshold": float(threshold),
                "metrics": binary_metrics(y_val, val_score, threshold),
            }
        else:
            best = tune_ensemble(p_bin_val, p_multi_val, y_val, betas, thresholds, min_sensitivity, objective)
            best["mode"] = "tuned"
            beta = best["beta"]
            threshold = best["threshold"]
        test_score = beta * p_bin_test + (1.0 - beta) * p_multi_test
        test_metrics = binary_metrics(y_test, test_score, threshold)

        fold_dir = ensemble_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_ensemble_predictions(fold_dir / "test_predictions_ensemble.csv", test_ids, y_test, p_bin_test, p_multi_test, beta, threshold)
        with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"selected": best, "test": test_metrics}, f, ensure_ascii=False, indent=2)

        row = {
            "fold": fold,
            "selected.mode": best["mode"],
            "selected.beta": beta,
            "selected.threshold": threshold,
            "selected.val_sensitivity": best["metrics"]["sensitivity"],
            "selected.val_specificity": best["metrics"]["specificity"],
        }
        for key, value in test_metrics.items():
            row[f"test.{key}"] = value
        fold_rows.append(row)

        flattened = {}
        flatten_numeric("selected", best, flattened)
        flatten_numeric("test", test_metrics, flattened)
        summary_rows.append(flattened)

    summary = summarize_metric_rows(summary_rows)
    with open(ensemble_dir / "summary_5fold.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_summary_csv(ensemble_dir / "summary_5fold.csv", summary)
    with open(ensemble_dir / "fold_settings.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = sorted(set().union(*(row.keys() for row in fold_rows)))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fold_rows)
    print_key_summary("small_ensemble", summary)
    return summary


def run_small_risk_score(output_dir, folds, alphas, thresholds, min_sensitivity, objective):
    risk_dir = output_dir / "small_risk_score"
    risk_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    summary_rows = []

    for fold in range(folds):
        multi_dir = output_dir / "small_multiclass" / f"fold_{fold}"
        required = [
            multi_dir / "val_predictions_best_auc.csv",
            multi_dir / "test_predictions_best_auc.csv",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            print(f"Skip small risk score: missing prediction files for fold {fold}.")
            return None

        val_ids, y_val, p_i63_val, p_other_val = align_risk_predictions(
            read_risk_predictions(multi_dir / "val_predictions_best_auc.csv")
        )
        test_ids, y_test, p_i63_test, p_other_test = align_risk_predictions(
            read_risk_predictions(multi_dir / "test_predictions_best_auc.csv")
        )
        best = tune_risk_score(p_i63_val, p_other_val, y_val, alphas, thresholds, min_sensitivity, objective)
        alpha = best["alpha"]
        threshold = best["threshold"]
        test_score = p_i63_test + alpha * p_other_test
        test_metrics = binary_metrics(y_test, test_score, threshold)

        fold_dir = risk_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_risk_predictions(fold_dir / "test_predictions_risk_score.csv", test_ids, y_test, p_i63_test, p_other_test, alpha, threshold)
        with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"selected": best, "test": test_metrics}, f, ensure_ascii=False, indent=2)

        row = {
            "fold": fold,
            "selected.alpha": alpha,
            "selected.threshold": threshold,
            "selected.val_sensitivity": best["metrics"]["sensitivity"],
            "selected.val_specificity": best["metrics"]["specificity"],
        }
        for key, value in test_metrics.items():
            row[f"test.{key}"] = value
        fold_rows.append(row)

        flattened = {}
        flatten_numeric("selected", best, flattened)
        flatten_numeric("test", test_metrics, flattened)
        summary_rows.append(flattened)

    summary = summarize_metric_rows(summary_rows)
    with open(risk_dir / "summary_5fold.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_summary_csv(risk_dir / "summary_5fold.csv", summary)
    with open(risk_dir / "fold_settings.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = sorted(set().union(*(row.keys() for row in fold_rows)))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fold_rows)
    print_key_summary("small_risk_score", summary)
    return summary


def print_key_summary(name, summary):
    keys = [
        "test.accuracy",
        "test.f1",
        "test.f1_macro",
        "test.f1_weighted",
        "test.auc",
        "test.sensitivity",
        "test.specificity",
        "test.balanced_accuracy",
        "test.binary_i63.accuracy",
        "test.binary_i63.f1",
        "test.binary_i63.auc",
        "test.binary_i63.sensitivity",
        "test.binary_i63.specificity",
    ]
    print("\n" + "-" * 80)
    print(f"5-fold mean ± std: {name}")
    for key in keys:
        stats = summary.get(key)
        if stats:
            print(f"{key}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    for key in sorted(summary):
        if key.startswith("binary_threshold_sweep.test."):
            stats = summary[key]
            print(f"{key}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    for key in ["selected.alpha", "selected.beta", "selected.threshold", "selected.metrics.sensitivity", "selected.metrics.specificity"]:
        stats = summary.get(key)
        if stats:
            print(f"{key}: {stats['mean']:.3f} ± {stats['std']:.3f}")


def aggregate_experiment(output_dir, experiment_name, folds):
    rows = []
    missing = []
    for fold in range(folds):
        metrics_path = output_dir / experiment_name / f"fold_{fold}" / "metrics.json"
        if not metrics_path.exists():
            missing.append(str(metrics_path))
            continue
        flattened = {}
        flatten_numeric("", load_json(metrics_path), flattened)
        rows.append(flattened)

    if missing:
        print(f"Skip summary for {experiment_name}: missing {len(missing)} metrics files.")
        return None
    if not rows:
        return None

    summary = summarize_metric_rows(rows)
    summary_dir = output_dir / experiment_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / "summary_5fold.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_summary_csv(summary_dir / "summary_5fold.csv", summary)
    print_key_summary(experiment_name, summary)
    return summary


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
    excel_root = args.excel_root.rstrip("/\\")
    data = ",".join(f"{excel_root}/{name}" for name in experiment["data"].split(","))
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
        "binary",
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
            )
        if not args.dry_run:
            aggregate_experiment(output_dir, experiment["name"], args.folds)

    selected_names = {experiment["name"] for experiment in experiments}
    if not args.dry_run and not args.no_ensemble and {"small_binary", "small_multiclass"}.issubset(selected_names):
        betas = parse_float_grid(args.ensemble_betas, np.arange(0.0, 1.0001, 0.05))
        thresholds = parse_float_grid(args.ensemble_thresholds, np.arange(0.20, 0.8001, 0.01))
        run_small_ensemble(
            output_dir,
            args.folds,
            args.ensemble_mode,
            args.ensemble_beta,
            args.ensemble_threshold,
            betas,
            thresholds,
            args.ensemble_min_sensitivity,
            args.ensemble_objective,
        )
    if not args.dry_run and not args.no_risk_score and "small_multiclass" in selected_names:
        alphas = parse_float_grid(args.risk_alphas, None)
        thresholds = parse_float_grid(args.risk_thresholds, None)
        run_small_risk_score(
            output_dir,
            args.folds,
            alphas,
            thresholds,
            args.risk_min_sensitivity,
            args.risk_objective,
        )


if __name__ == "__main__":
    main()
