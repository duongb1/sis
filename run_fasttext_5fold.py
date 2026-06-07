import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from sislib.data.labels import EXCEL_MULTICLASS_LABEL_MAP, EXCEL_TEXT_COLUMNS, binary_i63_from_multiclass, normalize_multiclass_label
from sislib.data.splits import assign_kfold_splits


TEXT_FIELDS_ALL = EXCEL_TEXT_COLUMNS
TEXT_FIELDS_CHIEF_EXAM = [
    "LYDO",
    "HB_BENHLY",
    "KB_TOANTHAN",
    "KB_BOPHAN",
]

VALID_MULTICLASS_LABELS = {
    "DISTANT_OTHER",
    "I63_INFARCTION",
    "OTHER_STROKE_LIKE",
}
MULTICLASS_LABELS = ["DISTANT_OTHER", "I63_INFARCTION", "OTHER_STROKE_LIKE"]
BINARY_MAPPING = {
    "I63_INFARCTION": "co",
    "OTHER_STROKE_LIKE": "khong",
    "DISTANT_OTHER": "khong",
}
POSITIVE_MULTICLASS_LABEL = "I63_INFARCTION"
POSITIVE_BINARY_LABEL = "co"

MODELS = [
    {
        "name": "fasttext_binary_all_fields",
        "task": "binary",
        "input_mode": "all_fields",
        "fields": TEXT_FIELDS_ALL,
    },
    {
        "name": "fasttext_binary_chief_exam",
        "task": "binary",
        "input_mode": "chief_exam",
        "fields": TEXT_FIELDS_CHIEF_EXAM,
    },
    {
        "name": "fasttext_multiclass_to_binary_all_fields",
        "task": "multiclass_to_binary",
        "input_mode": "all_fields",
        "fields": TEXT_FIELDS_ALL,
    },
    {
        "name": "fasttext_multiclass_to_binary_chief_exam",
        "task": "multiclass_to_binary",
        "input_mode": "chief_exam",
        "fields": TEXT_FIELDS_CHIEF_EXAM,
    },
]

PHOBERT_ATTNPOOL_REFERENCE = {
    "accuracy": 0.799,
    "f1": 0.811,
    "auc": 0.866,
    "sensitivity": 0.840,
    "specificity": 0.757,
    "balanced_accuracy": 0.799,
    "fp": 167,
    "fn": 114,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run FastText 5-fold baselines for small SIS Excel text classification.")
    parser.add_argument("--excel-root", default="/kaggle/input/datasets/duongbui/siscth")
    parser.add_argument("--output-dir", default="/kaggle/working/sis_excel_5fold_fasttext_mcstrat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--word-ngrams", type=int, default=2)
    parser.add_argument("--minn", type=int, default=2)
    parser.add_argument("--maxn", type=int, default=5)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--loss", default="softmax")
    parser.add_argument("--thread", type=int, default=4)
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--autotune-duration", type=int, default=120)
    parser.add_argument("--force", action="store_true", help="Overwrite existing fold output files.")
    return parser.parse_args()


def normalize_text(s: str) -> str:
    if s is None or pd.isna(s):
        return ""
    text = str(s).lower()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_text(row, fields):
    parts = []
    for field in fields:
        value = normalize_text(row.get(field, ""))
        if value:
            parts.append(f"[{field}] {value}")
    return " ".join(parts)


def fasttext_escape(text):
    return re.sub(r"\s+", " ", str(text).replace("\n", " ").replace("\t", " ")).strip()


def parse_thresholds(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def excel_paths(excel_root):
    root = Path(excel_root)
    return [
        root / "700_co_label.xlsx",
        root / "700_khong_label.xlsx",
    ]


def load_excel(excel_root):
    frames = []
    for path in excel_paths(excel_root):
        if not path.exists():
            raise FileNotFoundError(f"Missing required Excel file: {path}")
        df = pd.read_excel(path)
        missing = [field for field in [*TEXT_FIELDS_ALL, "LABEL"] if field not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        df["source_file"] = path.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    raw_labels = set(df["LABEL"].dropna().astype(str).unique())
    unexpected_raw = raw_labels - set(EXCEL_MULTICLASS_LABEL_MAP)
    if unexpected_raw:
        raise ValueError(f"Unexpected raw LABEL values before 3-class mapping: {unexpected_raw}")

    df["raw_LABEL"] = df["LABEL"]
    df["LABEL"] = df["LABEL"].map(normalize_multiclass_label)
    bad_labels = set(df["LABEL"].dropna().unique()) - VALID_MULTICLASS_LABELS
    if bad_labels:
        raise ValueError(f"Unexpected LABEL values: {bad_labels}")

    print("Multiclass distribution:")
    print(df["LABEL"].value_counts())
    return df.reset_index(drop=True)


def make_splits(df, args):
    records = [
        {
            "index": index,
            "label": int(binary_i63_from_multiclass(row["LABEL"])),
            "binary_label_name": BINARY_MAPPING[row["LABEL"]],
            "multiclass_label_name": row["LABEL"],
        }
        for index, row in df.iterrows()
    ]
    for fold in range(args.folds):
        split_records = assign_kfold_splits(
            [dict(record) for record in records],
            seed=args.seed,
            n_folds=args.folds,
            fold_index=fold,
            val_ratio=args.val_ratio,
            split_label="multiclass",
        )
        indices_by_split = {"train": [], "val": [], "test": []}
        for record in split_records:
            indices_by_split[record["split"]].append(record["index"])
        yield (
            fold,
            df.iloc[indices_by_split["train"]].reset_index(drop=True),
            df.iloc[indices_by_split["val"]].reset_index(drop=True),
            df.iloc[indices_by_split["test"]].reset_index(drop=True),
        )


def training_label(row, task):
    if task == "binary":
        return BINARY_MAPPING[row["LABEL"]]
    if task == "multiclass_to_binary":
        return row["LABEL"]
    raise ValueError(f"Unsupported task: {task}")


def write_fasttext_file(path, df, fields, task):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for _, row in df.iterrows():
            label = training_label(row, task)
            text = fasttext_escape(build_text(row, fields))
            f.write(f"__label__{label} {text}\n")


def train_fasttext(train_path, val_path, args):
    try:
        import fasttext
    except ImportError as exc:
        raise ImportError("Python package 'fasttext' is required. Install it with: pip install fasttext") from exc

    if args.autotune:
        return fasttext.train_supervised(
            input=str(train_path),
            autotuneValidationFile=str(val_path),
            autotuneDuration=args.autotune_duration,
            verbose=0,
        )
    return fasttext.train_supervised(
        input=str(train_path),
        lr=args.lr,
        epoch=args.epoch,
        wordNgrams=args.word_ngrams,
        minn=args.minn,
        maxn=args.maxn,
        dim=args.dim,
        loss=args.loss,
        thread=args.thread,
        verbose=0,
    )


def get_label_probs(model, text: str) -> dict:
    try:
        labels, probs = model.predict(text, k=-1)
    except ValueError as exc:
        if "Unable to avoid copy" not in str(exc) or not hasattr(model, "f"):
            raise
        predictions = model.f.predict(text, k=-1, threshold=0.0, on_unicode_error="strict")
        if predictions:
            probs, labels = zip(*predictions)
        else:
            labels, probs = [], []
    return {label: float(prob) for label, prob in zip(labels, probs)}


def predict_binary(model, df, fields, task, threshold):
    y_true, y_prob, y_pred = [], [], []
    for _, row in df.iterrows():
        text = fasttext_escape(build_text(row, fields))
        probs = get_label_probs(model, text)
        if task == "binary":
            p_i63 = probs.get(f"__label__{POSITIVE_BINARY_LABEL}", 0.0)
        elif task == "multiclass_to_binary":
            p_i63 = probs.get(f"__label__{POSITIVE_MULTICLASS_LABEL}", 0.0)
        else:
            raise ValueError(f"Unsupported task: {task}")
        y_true.append(1 if row["LABEL"] == POSITIVE_MULTICLASS_LABEL else 0)
        y_prob.append(p_i63)
        y_pred.append(1 if p_i63 >= threshold else 0)
    return np.asarray(y_true), np.asarray(y_prob), np.asarray(y_pred)


def binary_metrics(y_true, y_prob, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cm": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def threshold_sweep(model, df, fields, task, thresholds):
    sweep = {}
    for threshold in thresholds:
        y_true, y_prob, y_pred = predict_binary(model, df, fields, task, threshold)
        sweep[f"{threshold:g}"] = binary_metrics(y_true, y_prob, y_pred)
    return sweep


def hyperparameters(args):
    return {
        "epoch": args.epoch,
        "lr": args.lr,
        "word_ngrams": args.word_ngrams,
        "minn": args.minn,
        "maxn": args.maxn,
        "dim": args.dim,
        "loss": args.loss,
        "autotune": bool(args.autotune),
        "autotune_duration": args.autotune_duration if args.autotune else None,
    }


def label_schema():
    return {
        "multiclass_labels": MULTICLASS_LABELS,
        "binary_mapping": BINARY_MAPPING,
        "positive_label": POSITIVE_MULTICLASS_LABEL,
    }


def save_metrics(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, ensure_ascii=False, indent=2, allow_nan=False)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def mean_std(values):
    arr = np.asarray([value for value in values if not math.isnan(value)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def format_mean_std(values):
    mean, std = mean_std(values)
    if math.isnan(mean):
        return "nan"
    return f"{mean:.3f}±{std:.3f}"


def aggregate_cm(metrics_rows):
    tn = sum(row["test"]["tn"] for row in metrics_rows)
    fp = sum(row["test"]["fp"] for row in metrics_rows)
    fn = sum(row["test"]["fn"] for row in metrics_rows)
    tp = sum(row["test"]["tp"] for row in metrics_rows)
    return tn, fp, fn, tp


def print_threshold_table(model_name, fold, sweep):
    print(f"\nThreshold sweep: {model_name} fold {fold}")
    print("threshold | acc | f1 | auc | sens | spec | bal_acc")
    for threshold, metrics in sweep.items():
        auc = metrics["auc"]
        auc_text = "nan" if math.isnan(auc) else f"{auc:.3f}"
        print(
            f"{threshold:>9} | {metrics['accuracy']:.3f} | {metrics['f1']:.3f} | "
            f"{auc_text} | {metrics['sensitivity']:.3f} | {metrics['specificity']:.3f} | "
            f"{metrics['balanced_accuracy']:.3f}"
        )


def print_summary(results_by_model):
    print("\n5-fold summary: FastText small models\n")
    print(f"{'Model':48} {'Acc':9} {'F1':9} {'AUC':9} {'Sens':9} {'Spec':9} {'BalAcc':9} FP/FN")
    for model_name, rows in results_by_model.items():
        tn, fp, fn, tp = aggregate_cm(rows)
        print(
            f"{model_name:48} "
            f"{format_mean_std([row['test']['accuracy'] for row in rows]):9} "
            f"{format_mean_std([row['test']['f1'] for row in rows]):9} "
            f"{format_mean_std([row['test']['auc'] for row in rows]):9} "
            f"{format_mean_std([row['test']['sensitivity'] for row in rows]):9} "
            f"{format_mean_std([row['test']['specificity'] for row in rows]):9} "
            f"{format_mean_std([row['test']['balanced_accuracy'] for row in rows]):9} "
            f"{fp}/{fn}"
        )

    print("\nAggregate confusion counts:")
    for model_name, rows in results_by_model.items():
        tn, fp, fn, tp = aggregate_cm(rows)
        print(f"{model_name:48} TN={tn} FP={fp} FN={fn} TP={tp}")

    flat = [(model_name, row) for model_name, rows in results_by_model.items() for row in rows]
    best_specs = [
        ("best_f1", "f1", max),
        ("best_auc", "auc", max),
        ("best_sensitivity", "sensitivity", max),
        ("best_specificity", "specificity", max),
        ("best_balanced_accuracy", "balanced_accuracy", max),
        ("lowest_fn", "fn", min),
        ("lowest_fp", "fp", min),
    ]
    print("\nBest metrics:")
    for display_name, key, selector in best_specs:
        candidates = [(model_name, row) for model_name, row in flat if not math.isnan(float(row["test"][key]))]
        if not candidates:
            print(f"{display_name}: nan")
            continue
        model_name, row = selector(candidates, key=lambda item: item[1]["test"][key])
        print(f"{display_name}: {row['test'][key]} ({model_name}, fold {row['fold']})")

    ref = PHOBERT_ATTNPOOL_REFERENCE
    print("\nPhoBERT-AttnPool reference baseline:")
    print(
        f"Acc={ref['accuracy']:.3f} F1={ref['f1']:.3f} AUC={ref['auc']:.3f} "
        f"Sens={ref['sensitivity']:.3f} Spec={ref['specificity']:.3f} "
        f"BalAcc={ref['balanced_accuracy']:.3f} FP/FN={ref['fp']}/{ref['fn']}"
    )


def run_model_fold(model_cfg, fold, train_df, val_df, test_df, output_dir, args, thresholds):
    fold_dir = output_dir / model_cfg["name"] / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_path = fold_dir / "train.txt"
    val_path = fold_dir / "val.txt"
    test_path = fold_dir / "test.txt"
    model_path = fold_dir / "model.bin"
    metrics_path = fold_dir / "metrics.json"

    if metrics_path.exists() and model_path.exists() and not args.force:
        raise FileExistsError(f"Output exists, use --force to overwrite: {fold_dir}")

    write_fasttext_file(train_path, train_df, model_cfg["fields"], model_cfg["task"])
    write_fasttext_file(val_path, val_df, model_cfg["fields"], model_cfg["task"])
    write_fasttext_file(test_path, test_df, model_cfg["fields"], model_cfg["task"])

    model = train_fasttext(train_path, val_path, args)
    model.save_model(str(model_path))

    y_true, y_prob, y_pred = predict_binary(model, test_df, model_cfg["fields"], model_cfg["task"], args.threshold)
    test_metrics = binary_metrics(y_true, y_prob, y_pred)
    sweep = threshold_sweep(model, test_df, model_cfg["fields"], model_cfg["task"], thresholds)

    payload = {
        "model": model_cfg["name"],
        "task": model_cfg["task"],
        "input_mode": model_cfg["input_mode"],
        "fold": fold,
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "fields": model_cfg["fields"],
        "label_schema": label_schema(),
        "hyperparameters": hyperparameters(args),
        "test": test_metrics,
        "threshold_sweep": {
            "test": sweep,
        },
    }
    save_metrics(metrics_path, payload)
    print_threshold_table(model_cfg["name"], fold, sweep)
    return payload


def main():
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_excel(args.excel_root)
    results_by_model = {model_cfg["name"]: [] for model_cfg in MODELS}

    for fold, train_df, val_df, test_df in make_splits(df, args):
        print(f"\nFold {fold}: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
        for model_cfg in MODELS:
            print(f"\nTraining {model_cfg['name']} fold {fold}")
            payload = run_model_fold(model_cfg, fold, train_df, val_df, test_df, output_dir, args, thresholds)
            results_by_model[model_cfg["name"]].append(payload)

    print_summary(results_by_model)


if __name__ == "__main__":
    main()
