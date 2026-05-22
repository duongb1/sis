import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sislib.common import get_device, resolve_max_len, round_float, round_metrics, seed_all, to_device, unwrap
from sislib.metrics import cls_metrics
from sislib.mri import collect_pairs
from sislib.mri_teacher import compute_mri_logits, load_mri_teacher
from sislib.text_data import collect_paired_text
from sislib.text_train import ce_epoch, eval_text


LABEL_MAP = {"khong": 0, "không": 0, "0": 0, 0: 0, "co": 1, "có": 1, "1": 1, 1: 1}


def parse_args():
    p = argparse.ArgumentParser(description="Run OOF stacking meta-classifier for paired text classification.")
    p.add_argument("--images", default=None, help="Optional paired image root. If set, paired split CSVs are generated from patient txt files.")
    p.add_argument("--paired_train_csv", default=None)
    p.add_argument("--paired_val_csv", default=None)
    p.add_argument("--paired_test_csv", default=None)
    p.add_argument("--large_text_ckpt", required=True)
    p.add_argument("--paired_text_model_name_or_ckpt", required=True)
    p.add_argument("--mri_teacher_pred_csv", default=None)
    p.add_argument("--mri_teacher_dir", default=None, help="Optional train_mri.py output dir containing *_teacher_outputs_best_auc.csv.")
    p.add_argument("--mri_teacher_ckpt", default=None, help="Optional MRI teacher checkpoint. Defaults to mri_teacher_dir/best_auc_model.pt.")
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


class TextFrameDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row["text"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
        item["id"] = str(row["sample_id"])
        return item


def load_split(path):
    df = pd.read_csv(path)
    if "sample_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "sample_id"})
    required = {"sample_id", "text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df[["sample_id", "text", "label"]].copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["label"] = df["label"].map(lambda x: LABEL_MAP.get(x, LABEL_MAP.get(str(x).strip().lower(), x)))
    df["label"] = df["label"].astype(int)
    if df["sample_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicated sample_id.")
    return df


def records_to_frame(records):
    return pd.DataFrame(
        [
            {"sample_id": row["id"], "text": row["text"], "label": row["label"]}
            for row in records
        ]
    )


def load_splits(args):
    if args.images:
        records, missing, _ = collect_paired_text(args.images)
        if missing:
            missing_path = Path(args.output_dir) / "missing_paired_text_files.txt"
            missing_path.write_text("\n".join(missing), encoding="utf-8")
            print(f"Missing paired text files: {len(missing)} | saved {missing_path}")
        frames = {}
        for split in ["train", "val", "test"]:
            rows = [row for row in records if row["split"] == split]
            frames[split] = records_to_frame(rows)
            frames[split].to_csv(Path(args.output_dir) / f"paired_{split}.csv", index=False)
        return frames["train"], frames["val"], frames["test"]

    missing = [name for name, value in [
        ("--paired_train_csv", args.paired_train_csv),
        ("--paired_val_csv", args.paired_val_csv),
        ("--paired_test_csv", args.paired_test_csv),
    ] if not value]
    if missing:
        raise ValueError(f"Provide --images or all paired CSV arguments. Missing: {', '.join(missing)}")
    return load_split(args.paired_train_csv), load_split(args.paired_val_csv), load_split(args.paired_test_csv)


def normalize_mri_predictions(mri):
    if "sample_id" not in mri.columns and "id" in mri.columns:
        mri = mri.rename(columns={"id": "sample_id"})
    if "p_mri" not in mri.columns and "prob_co" in mri.columns:
        mri = mri.rename(columns={"prob_co": "p_mri"})
    if not {"sample_id", "p_mri"}.issubset(mri.columns):
        raise ValueError("MRI predictions must contain sample_id,p_mri or id,prob_co columns.")
    mri = mri[["sample_id", "p_mri"]].copy()
    mri["sample_id"] = mri["sample_id"].astype(str)
    mri["p_mri"] = mri["p_mri"].astype(float)
    if mri["sample_id"].duplicated().any():
        raise ValueError("MRI predictions contain duplicated sample_id.")
    return mri


def frames_to_records(train, val, test):
    records = []
    for df in [train, val, test]:
        for row in df.itertuples(index=False):
            records.append({"id": str(row.sample_id), "label": int(row.label)})
    return records


def teacher_logits_to_mri_frame(teacher_logits):
    rows = []
    for sample_id, logits in teacher_logits.items():
        values = np.asarray(logits, dtype=np.float64)
        exp_values = np.exp(values - values.max())
        prob_co = float(exp_values[1] / exp_values.sum())
        rows.append({"sample_id": str(sample_id), "p_mri": prob_co})
    return pd.DataFrame(rows)


def regenerate_mri_predictions(args, train, val, test, device):
    if not args.images:
        raise FileNotFoundError(
            "MRI teacher output CSVs are missing. Pass --images so p_mri can be regenerated from the MRI checkpoint."
        )
    ckpt = Path(args.mri_teacher_ckpt) if args.mri_teacher_ckpt else None
    if ckpt is None and args.mri_teacher_dir:
        ckpt = Path(args.mri_teacher_dir) / "best_auc_model.pt"
    if ckpt is None or not ckpt.exists():
        raise FileNotFoundError(
            f"MRI teacher output CSVs are missing and MRI checkpoint was not found: {ckpt}. "
            "Rerun train_mri.py or pass --mri_teacher_pred_csv."
        )

    records = frames_to_records(train, val, test)
    mri_records, image_root = collect_pairs(args.images)
    print(f"Regenerating MRI teacher predictions from {ckpt}")
    print(f"MRI image root: {image_root.resolve()}")
    teacher = load_mri_teacher(ckpt, device, not args.no_mgpu)
    teacher_logits = compute_mri_logits(records, mri_records, teacher, device, args.batch_mri, args.workers)
    mri = teacher_logits_to_mri_frame(teacher_logits)
    out_path = Path(args.output_dir) / "mri_teacher_predictions_merged.csv"
    mri.to_csv(out_path, index=False)
    print(f"Saved regenerated MRI teacher predictions: {out_path}")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return normalize_mri_predictions(mri)


def load_mri_predictions(args, train, val, test, device):
    if args.mri_teacher_pred_csv:
        mri = pd.read_csv(args.mri_teacher_pred_csv)
    elif args.mri_teacher_dir:
        root = Path(args.mri_teacher_dir)
        parts = []
        missing_paths = []
        for split in ["train", "val", "test"]:
            path = root / f"{split}_teacher_outputs_best_auc.csv"
            if not path.exists():
                missing_paths.append(path)
                continue
            parts.append(pd.read_csv(path))
        if missing_paths:
            print("Missing MRI teacher output CSVs:")
            for path in missing_paths:
                print(f"- {path}")
            return regenerate_mri_predictions(args, train, val, test, device)
        mri = pd.concat(parts, axis=0, ignore_index=True)
        mri.to_csv(Path(args.output_dir) / "mri_teacher_predictions_merged.csv", index=False)
    else:
        raise ValueError("Provide --mri_teacher_pred_csv or --mri_teacher_dir.")

    return normalize_mri_predictions(mri)


def load_data(args, device):
    train, val, test = load_splits(args)
    mri = load_mri_predictions(args, train, val, test, device)
    return train, val, test, mri


def validate_splits(train, val, test, mri):
    ids_train, ids_val, ids_test = set(train.sample_id), set(val.sample_id), set(test.sample_id)
    assert ids_train.isdisjoint(ids_val), "train and val sample_id overlap."
    assert ids_train.isdisjoint(ids_test), "train and test sample_id overlap."
    assert ids_val.isdisjoint(ids_test), "val and test sample_id overlap."
    all_ids = ids_train | ids_val | ids_test
    missing_mri = sorted(all_ids - set(mri.sample_id))
    print(f"MRI merge missing sample count: {len(missing_mri)}")
    assert not missing_mri, "mri_teacher_pred_csv does not cover all train/val/test sample_id."


def merge_mri(df, mri):
    out = df.merge(mri, on="sample_id", how="left")
    missing = int(out["p_mri"].isna().sum())
    print(f"MRI missing after merge for {len(df)} rows: {missing}")
    assert missing == 0, "Missing p_mri after merge."
    return out


def train_text_model_fold(train_df, model_name, output_dir, args, device, val_df=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "khong", 1: "co"},
        label2id={"khong": 0, "co": 1},
    )
    max_len = resolve_max_len(model, args.max_len)
    model = to_device(model, device, not args.no_mgpu)
    train_loader = DataLoader(
        TextFrameDataset(train_df, tokenizer, max_len),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, int(np.ceil(len(train_loader) / max(args.accum, 1))) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * args.warmup), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_score = -1.0
    best_dir = output_dir / "best_model"
    val_loader = None
    if val_df is not None:
        val_loader = DataLoader(
            TextFrameDataset(val_df, tokenizer, max_len),
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )

    for epoch in range(1, args.epochs + 1):
        loss = ce_epoch(model, train_loader, optimizer, scheduler, scaler, device, args.accum)
        if val_loader is None:
            print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={loss:.3f}")
            continue
        metrics, _, _, _, _ = eval_text(model, val_loader, device, args.threshold, desc="Validating")
        score = metrics["auc"]
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={loss:.3f} | val_auc={score:.3f} | val_f1={metrics['f1']:.3f}")
        if score > best_score:
            best_score = score
            unwrap(model).save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

    if val_loader is None:
        unwrap(model).save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_dir


@torch.no_grad()
def predict_text_model(model_name_or_dir, df, args, device, desc="Predicting"):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_dir)
    max_len = resolve_max_len(model, args.max_len)
    model = to_device(model, device, not args.no_mgpu)
    loader = DataLoader(
        TextFrameDataset(df, tokenizer, max_len),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    metrics, ids, labels, probs, preds = eval_text(model, loader, device, args.threshold, desc=desc)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame({"sample_id": ids, "label": labels, "prob": probs, "pred_05": preds}), metrics


def generate_oof_predictions(train, args, device, output_dir):
    y = train["label"].values
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    oof = []
    for fold, (tr_idx, ho_idx) in enumerate(skf.split(train, y), start=1):
        print(f"\nOOF fold {fold}/{args.n_folds}")
        fold_dir = Path(output_dir) / f"fold_{fold}"
        fold_model = train_text_model_fold(
            train.iloc[tr_idx].reset_index(drop=True),
            args.paired_text_model_name_or_ckpt,
            fold_dir,
            args,
            device,
            val_df=None,
        )
        pred, _ = predict_text_model(fold_model, train.iloc[ho_idx].reset_index(drop=True), args, device, desc=f"OOF fold {fold}")
        pred = pred.rename(columns={"prob": "p_paired_oof"})
        oof.append(pred[["sample_id", "label", "p_paired_oof"]])
    out = pd.concat(oof, axis=0).sort_values("sample_id").reset_index(drop=True)
    assert len(out) == len(train), "OOF row count mismatch."
    assert not out["p_paired_oof"].isna().any(), "OOF contains missing p_paired_oof."
    return out


def build_meta_features(df, use_mri):
    if use_mri:
        cols = ["p_paired", "p_large", "p_mri", "abs_p_paired_05", "abs_p_large_05", "p_large_minus_paired"]
    else:
        cols = ["p_paired", "p_large"]
    return df[cols].values.astype(np.float32)


def add_derived_features(df):
    out = df.copy()
    out["abs_p_paired_05"] = (out["p_paired"] - 0.5).abs()
    out["abs_p_large_05"] = (out["p_large"] - 0.5).abs()
    out["p_large_minus_paired"] = out["p_large"] - out["p_paired"]
    return out


def train_logistic_meta(train_df, val_df, use_mri):
    x_train = build_meta_features(train_df, use_mri)
    y_train = train_df["label"].values.astype(int)
    x_val = build_meta_features(val_df, use_mri)
    y_val = val_df["label"].values.astype(int)
    best = None
    for class_weight in [None, "balanced"]:
        for c_value in [0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(penalty="l2", C=c_value, class_weight=class_weight, max_iter=1000, solver="liblinear")
            clf.fit(x_train, y_train)
            val_prob = clf.predict_proba(x_val)[:, 1]
            val_auc = cls_metrics(y_val, val_prob, (val_prob >= 0.5).astype(int))["auc"]
            cand = {"model": clf, "C": c_value, "class_weight": class_weight, "val_auc": val_auc, "use_mri": use_mri}
            if best is None or val_auc > best["val_auc"]:
                best = cand
    return best


def select_threshold(labels, probs, objective):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    best_t, best_score = 0.5, -1.0
    for threshold in np.linspace(0.0, 1.0, 1001):
        preds = (probs >= threshold).astype(int)
        m = cls_metrics(labels, probs, preds, threshold=float(threshold))
        if objective == "f1":
            score = m["f1"]
        elif objective == "balanced_accuracy":
            score = balanced_accuracy_score(labels, preds)
        elif objective == "youden":
            score = m["sensitivity"] + m["specificity"] - 1.0
        else:
            raise ValueError(objective)
        if score > best_score:
            best_score, best_t = score, float(threshold)
    return best_t, best_score


def compute_metrics(labels, probs, threshold):
    preds = (np.asarray(probs) >= threshold).astype(int)
    m = cls_metrics(labels, probs, preds, threshold=threshold)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    m["confusion_matrix"] = [[int(cm[0, 0]), int(cm[0, 1])], [int(cm[1, 0]), int(cm[1, 1])]]
    m["balanced_accuracy"] = float(balanced_accuracy_score(labels, preds))
    return round_metrics(m)


def run_weighted_average(val_df, test_df, objective):
    best = None
    for alpha in [0.85, 0.90, 0.95, 0.97, 0.99]:
        val_prob = alpha * val_df["p_paired"].values + (1.0 - alpha) * val_df["p_large"].values
        threshold, score = select_threshold(val_df["label"].values, val_prob, objective)
        cand = {"alpha": alpha, "threshold": threshold, "score": score}
        if best is None or score > best["score"]:
            best = cand
    test_prob = best["alpha"] * test_df["p_paired"].values + (1.0 - best["alpha"]) * test_df["p_large"].values
    return best, test_prob


def save_prediction_frame(path, df):
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def result_row(model_name, split, labels, probs, threshold, extra):
    row = {
        "model": model_name,
        "split": split,
        **extra,
        **compute_metrics(labels, probs, threshold),
    }
    return row


def save_outputs(output_dir, train_meta, val_meta, test_meta, results, best_meta):
    output_dir = Path(output_dir)
    save_prediction_frame(output_dir / "train_oof_predictions.csv", train_meta)
    save_prediction_frame(output_dir / "val_predictions.csv", val_meta)
    save_prediction_frame(output_dir / "test_predictions.csv", test_meta)
    with open(output_dir / "stacking_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    pd.DataFrame(results).to_csv(output_dir / "stacking_results.csv", index=False)
    joblib.dump(best_meta, output_dir / "best_meta_model.joblib")
    with open(output_dir / "README_stacking_run.md", "w", encoding="utf-8") as f:
        f.write(
            "# Stacking Run\n\n"
            "- OOF predictions are generated only inside paired_train.\n"
            "- paired_val is used for alpha/C/class_weight/threshold selection.\n"
            "- paired_test is used only once for final reporting.\n"
            "- stack_with_mri uses p_mri at decision time and is MRI-assisted late fusion, not text-only deployment.\n"
        )


def main():
    args = parse_args()
    seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.cpu)

    train, val, test, mri = load_data(args, device)
    validate_splits(train, val, test, mri)
    train = merge_mri(train, mri)
    val = merge_mri(val, mri)
    test = merge_mri(test, mri)

    oof = generate_oof_predictions(train, args, device, output_dir / "oof_models")
    assert "p_paired_oof" in oof.columns, "train meta must use p_paired_oof."

    p_large_train, _ = predict_text_model(args.large_text_ckpt, train, args, device, desc="Large-text train")
    p_large_val, _ = predict_text_model(args.large_text_ckpt, val, args, device, desc="Large-text val")
    p_large_test, _ = predict_text_model(args.large_text_ckpt, test, args, device, desc="Large-text test")

    paired_dir = train_text_model_fold(train, args.paired_text_model_name_or_ckpt, output_dir / "paired_full_model", args, device, val_df=val)
    p_paired_val, _ = predict_text_model(paired_dir, val, args, device, desc="Paired final val")
    p_paired_test, _ = predict_text_model(paired_dir, test, args, device, desc="Paired final test")

    large_ft_dir = train_text_model_fold(train, args.large_text_ckpt, output_dir / "large_text_paired_ft_model", args, device, val_df=val)
    p_large_ft_val, _ = predict_text_model(large_ft_dir, val, args, device, desc="Large-text->paired val")
    p_large_ft_test, _ = predict_text_model(large_ft_dir, test, args, device, desc="Large-text->paired test")

    train_meta = train.merge(oof[["sample_id", "p_paired_oof"]], on="sample_id")
    train_meta = train_meta.merge(p_large_train[["sample_id", "prob"]].rename(columns={"prob": "p_large"}), on="sample_id")
    train_meta["p_paired"] = train_meta["p_paired_oof"]
    train_meta = add_derived_features(train_meta)
    assert "p_paired_oof" in train_meta.columns and np.allclose(train_meta["p_paired"], train_meta["p_paired_oof"])

    val_meta = val.merge(p_large_val[["sample_id", "prob"]].rename(columns={"prob": "p_large"}), on="sample_id")
    val_meta = val_meta.merge(p_paired_val[["sample_id", "prob"]].rename(columns={"prob": "p_paired"}), on="sample_id")
    val_meta = val_meta.merge(p_large_ft_val[["sample_id", "prob"]].rename(columns={"prob": "p_large_ft"}), on="sample_id")
    val_meta = add_derived_features(val_meta)

    test_meta = test.merge(p_large_test[["sample_id", "prob"]].rename(columns={"prob": "p_large"}), on="sample_id")
    test_meta = test_meta.merge(p_paired_test[["sample_id", "prob"]].rename(columns={"prob": "p_paired"}), on="sample_id")
    test_meta = test_meta.merge(p_large_ft_test[["sample_id", "prob"]].rename(columns={"prob": "p_large_ft"}), on="sample_id")
    test_meta = add_derived_features(test_meta)

    results = []
    for name, prob_col in [("Paired-only CE", "p_paired"), ("Large-text direct", "p_large"), ("Large-text -> paired CE", "p_large_ft")]:
        threshold, score = select_threshold(val_meta["label"].values, val_meta[prob_col].values, "f1")
        results.append(result_row(name, "test", test_meta["label"].values, test_meta[prob_col].values, threshold, {"selection": "val_f1", "val_score": round_float(score)}))

    for objective in ["f1", "balanced_accuracy"]:
        best_avg, test_prob = run_weighted_average(val_meta, test_meta, objective)
        results.append(result_row("Weighted average text-only", "test", test_meta["label"].values, test_prob, best_avg["threshold"], {"selection": f"val_{objective}", "alpha": best_avg["alpha"], "val_score": round_float(best_avg["score"])}))

    best_meta_to_save = None
    best_meta_score = -1.0
    for use_mri, model_name in [(False, "Stacking text-only"), (True, "Stacking with MRI")]:
        meta = train_logistic_meta(train_meta, val_meta, use_mri)
        x_val = build_meta_features(val_meta, use_mri)
        x_test = build_meta_features(test_meta, use_mri)
        val_prob = meta["model"].predict_proba(x_val)[:, 1]
        test_prob = meta["model"].predict_proba(x_test)[:, 1]
        for objective in ["f1", "balanced_accuracy", "youden"]:
            threshold, score = select_threshold(val_meta["label"].values, val_prob, objective)
            row = result_row(
                model_name,
                "test",
                test_meta["label"].values,
                test_prob,
                threshold,
                {
                    "selection": f"val_{objective}",
                    "C": meta["C"],
                    "class_weight": meta["class_weight"] if meta["class_weight"] is not None else "none",
                    "val_auc_for_meta_grid": round_float(meta["val_auc"]),
                    "val_score": round_float(score),
                },
            )
            results.append(row)
            if objective == "f1" and score > best_meta_score:
                best_meta_score = score
                best_meta_to_save = {"name": model_name, "threshold": threshold, **meta}

    save_outputs(output_dir, train_meta, val_meta, test_meta, results, best_meta_to_save)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
