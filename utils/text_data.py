import csv
import random
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split

from .common import LABELS, LABEL_TO_ID, SPLITS, read_text

EXCEL_TEXT_COLUMNS = ["LYDO", "HB_BENHLY", "HB_BANTHAN", "KB_TOANTHAN", "KB_BOPHAN"]


def assign_kfold_splits(records, seed=42, n_folds=5, fold_index=0, val_ratio=0.1, split_label="target"):
    if n_folds < 2:
        raise ValueError("--n-folds must be at least 2.")
    if fold_index < 0 or fold_index >= n_folds:
        raise ValueError(f"--fold-index must be between 0 and {n_folds - 1}.")

    if split_label == "binary":
        y = [record["binary_label_name"] for record in records]
    elif split_label == "target":
        y = [record["label"] for record in records]
    else:
        raise ValueError(f"Unknown split_label: {split_label}")

    indices = list(range(len(records)))
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(splitter.split(indices, y))
    train_val_idx, test_idx = folds[fold_index]
    train_val_idx = list(train_val_idx)
    test_idx = list(test_idx)

    test_ratio = 1.0 / n_folds
    val_ratio_within_train_val = val_ratio / (1.0 - test_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_within_train_val,
        random_state=seed + fold_index,
        stratify=[y[index] for index in train_val_idx],
    )
    split_by_index = {index: "train" for index in train_idx}
    split_by_index.update({index: "val" for index in val_idx})
    split_by_index.update({index: "test" for index in test_idx})
    for index, record in enumerate(records):
        record["split"] = split_by_index[index]
        record["fold"] = fold_index
    return records


def parse_labels_arg(value):
    if value is None:
        return None
    labels = [item.strip() for item in str(value).split(",") if item.strip()]
    return labels or None


def make_label_maps(labels):
    labels = list(labels)
    return {label: index for index, label in enumerate(labels)}, {index: label for index, label in enumerate(labels)}


def _csv_paths(data):
    paths = []
    for item in str(data).split(","):
        item = item.strip()
        if not item:
            continue
        path = Path(item)
        if path.is_file() and path.suffix.lower() == ".csv":
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(p for p in path.glob("*.csv") if not p.name.startswith("._")))
    return sorted(dict.fromkeys(paths))


def _clean_cell(value):
    if value is None:
        return ""
    try:
        import pandas as pd
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _csv_label_name(value):
    label = _clean_cell(value).lower()
    if "nhồi máu não cấp" in label or label in {"1", "1.0", "co", "yes", "true", "i63", "i63_infarction"}:
        return "co"
    if "bệnh khác" in label or label in {"0", "0.0", "khong", "không", "no", "false", "non_i63"}:
        return "khong"
    raise ValueError(f"Unsupported raw CSV binary label: {value!r}")


def _join_csv_row(row):
    parts = []
    for col, prefix in [
        ("LYDO", "Lý do vào viện:"),
        ("HB_BENHLY", "Bệnh sử hiện tại:"),
        ("HB_BANTHAN", "Tiền sử bản thân:"),
        ("KB_TOANTHAN", "Khám toàn thân:"),
        ("KB_BOPHAN", "Khám bộ phận:")
    ]:
        val = _clean_cell(row.get(col))
        if val:
            parts.append(f"{prefix} {val}")
    return "\n".join(parts).strip()


def _split_by_label(records, seed=42, val_ratio=0.1, test_ratio=0.1):
    grouped = {}
    for record in records:
        grouped.setdefault(record["label_name"], []).append(record)
    rng = random.Random(seed)
    for rows in grouped.values():
        rng.shuffle(rows)
        n = len(rows)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        if test_ratio > 0 and n >= 3:
            n_test = max(1, n_test)
        if val_ratio > 0 and n >= 3:
            n_val = max(1, n_val)
        if n_test + n_val >= n:
            overflow = n_test + n_val - max(n - 1, 0)
            n_val = max(0, n_val - overflow)
        for index, row in enumerate(rows):
            if index < n_test:
                row["split"] = "test"
            elif index < n_test + n_val:
                row["split"] = "val"
            else:
                row["split"] = "train"
    return records


def _assign_single_split(records, split_name="eval"):
    for record in records:
        record["split"] = split_name
    return records


def discover_csv_labels(data):
    labels = set()
    for path in _csv_paths(data):
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            label_col = None
            for col in reader.fieldnames or []:
                if col.upper() == "LABEL":
                    label_col = col
                    break
            if not label_col:
                continue
            for row in reader:
                val = _clean_cell(row.get(label_col))
                if val:
                    try:
                        lbl_name = _csv_label_name(val)
                        labels.add(lbl_name)
                    except ValueError:
                        pass
    return [label for label in ["khong", "co"] if label in labels]


def collect_csv_text(
    data,
    labels=None,
    label_to_id=None,
    seed=42,
    val_ratio=0.1,
    test_ratio=0.2,
    split_strategy="random",
    n_folds=5,
    fold_index=0,
    eval_split_name="eval",
):
    paths = _csv_paths(data)
    if not paths:
        raise FileNotFoundError(f"Cannot find CSV files under {data}.")
    labels = list(labels) if labels is not None else discover_csv_labels(data)
    if not labels:
        labels = ["khong", "co"]
    if label_to_id is None:
        label_to_id, _ = make_label_maps(labels)

    records, skipped = [], []
    for path in paths:
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            label_col = None
            for col in fieldnames:
                if col.upper() == "LABEL":
                    label_col = col
                    break
            if not label_col:
                raise ValueError(f"{path} is missing a LABEL column.")
            missing_columns = [column for column in ["LYDO", "HB_BENHLY", "HB_BANTHAN", "KB_TOANTHAN", "KB_BOPHAN"] if column not in fieldnames]
            if missing_columns:
                raise ValueError(f"{path} is missing required columns: {', '.join(missing_columns)}")
            for index, row in enumerate(reader, start=2):
                try:
                    label_name = _csv_label_name(row.get(label_col))
                except ValueError as exc:
                    skipped.append(f"{path}:{index}:{exc}")
                    continue
                if label_name not in label_to_id:
                    skipped.append(f"{path}:{index}:unknown_label:{label_name}")
                    continue
                text = _join_csv_row(row)
                if not text:
                    skipped.append(f"{path}:{index}:empty_text")
                    continue
                records.append(
                    {
                        "id": f"{path.stem}_{index:06d}",
                        "split": "",
                        "label": label_to_id[label_name],
                        "label_name": label_name,
                        "text_path": str(path),
                        "text": text,
                        "source_file": path.name,
                        "row_index": index,
                        "binary_label_name": label_name,
                    }
                )

    if split_strategy == "kfold":
        records = assign_kfold_splits(
            records,
            seed=seed,
            n_folds=n_folds,
            fold_index=fold_index,
            val_ratio=val_ratio,
            split_label="target",
        )
    elif split_strategy == "random":
        records = _split_by_label(records, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)
    elif split_strategy in {"eval", "none"}:
        records = _assign_single_split(records, split_name=eval_split_name)
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}")

    data_text = str(data)
    root = Path(data_text.split(",")[0]).parent if "," in data_text else Path(data_text).parent
    return records, skipped, root


class TextDataset(Dataset):
    def __init__(self, records, tokenizer, max_len, sample_weights=None):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sample_weights = sample_weights

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(row["label"], dtype=torch.long)
        if self.sample_weights is not None:
            item["sample_weight"] = torch.tensor(self.sample_weights[row["id"]], dtype=torch.float32)
        item["id"] = row["id"]
        return item


def save_records(path, records):
    optional_fields = ["source_file", "row_index", "binary_label_name", "fold"]
    fieldnames = ["id", "split", "label", "label_name", "text_path"]
    fieldnames.extend(field for field in optional_fields if any(field in row for row in records))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
