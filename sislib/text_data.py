import csv
import random
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split

from .common import LABELS, LABEL_TO_ID, SPLITS, read_text


EXCEL_TEXT_COLUMNS = ["LYDO", "HB_BENHLY", "HB_BANTHAN", "HB_GIADINH", "KB_TOANTHAN", "KB_BOPHAN"]


def parse_labels_arg(value):
    if value is None:
        return None
    labels = [item.strip() for item in str(value).split(",") if item.strip()]
    return labels or None


def make_label_maps(labels):
    labels = list(labels)
    return {label: index for index, label in enumerate(labels)}, {index: label for index, label in enumerate(labels)}


def _excel_paths(data):
    paths = []
    for item in str(data).split(","):
        item = item.strip()
        if not item:
            continue
        path = Path(item)
        if path.is_file() and path.suffix.lower() in {".xlsx", ".xls"}:
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(p for p in path.glob("*.xls*") if not p.name.startswith("~$")))
    return sorted(dict.fromkeys(paths))


def is_excel_data(data):
    return bool(_excel_paths(data))


def _read_excel(path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Reading Excel input requires pandas and openpyxl to be installed.") from exc
    return pd.read_excel(path)


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


def _label_from_excel_filename(path):
    tokens = re.split(r"[^a-z0-9]+", path.stem.lower())
    if "khong" in tokens:
        return "khong"
    if "co" in tokens:
        return "co"
    raise ValueError(f"Cannot infer binary label from Excel filename: {path.name}")


def _join_excel_row(row):
    parts = []
    for column in EXCEL_TEXT_COLUMNS:
        if column not in row:
            continue
        value = _clean_cell(row[column])
        if value:
            parts.append(value)
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


def _split_excel_kfold(records, seed=42, n_folds=5, fold_index=0, val_ratio=0.1):
    if n_folds < 2:
        raise ValueError("--n-folds must be at least 2.")
    if fold_index < 0 or fold_index >= n_folds:
        raise ValueError(f"--fold-index must be between 0 and {n_folds - 1}.")

    y = [record["label"] for record in records]
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


def discover_excel_labels(data, task="multiclass"):
    if task == "binary":
        found = {_label_from_excel_filename(path) for path in _excel_paths(data)}
        return [label for label in sorted(found, key=lambda item: LABEL_TO_ID[item])]
    labels = set()
    for path in _excel_paths(data):
        df = _read_excel(path)
        if "LABEL" not in df.columns:
            raise ValueError(f"Missing LABEL column in {path}")
        labels.update(_clean_cell(value) for value in df["LABEL"].tolist())
    return sorted(label for label in labels if label)


def collect_excel_text(
    data,
    labels=None,
    label_to_id=None,
    task="multiclass",
    seed=42,
    val_ratio=0.1,
    test_ratio=0.1,
    split_strategy="random",
    n_folds=5,
    fold_index=0,
):
    paths = _excel_paths(data)
    if not paths:
        raise FileNotFoundError(f"Cannot find Excel files under {data}.")
    labels = list(labels) if labels is not None else discover_excel_labels(data, task=task)
    if label_to_id is None:
        label_to_id, _ = make_label_maps(labels)

    records, skipped = [], []
    for path in paths:
        df = _read_excel(path)
        missing_columns = [column for column in [*EXCEL_TEXT_COLUMNS, "LABEL"] if column not in df.columns]
        if missing_columns:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing_columns)}")
        binary_label_name = _label_from_excel_filename(path)
        for index, row in df.iterrows():
            row_dict = row.to_dict()
            multiclass_label = _clean_cell(row_dict.get("LABEL"))
            label_name = binary_label_name if task == "binary" else multiclass_label
            if not label_name:
                skipped.append(f"{path}:{index + 2}:missing_label")
                continue
            if label_name not in label_to_id:
                skipped.append(f"{path}:{index + 2}:unknown_label:{label_name}")
                continue
            text = _join_excel_row(row_dict)
            if not text:
                skipped.append(f"{path}:{index + 2}:empty_text")
                continue
            records.append(
                {
                    "id": f"{path.stem}_{index + 2:06d}",
                    "split": "",
                    "label": label_to_id[label_name],
                    "label_name": label_name,
                    "text_path": str(path),
                    "text": text,
                    "source_file": path.name,
                    "row_index": index + 2,
                    "binary_label_name": binary_label_name,
                    "multiclass_label_name": multiclass_label,
                }
            )
    if split_strategy == "kfold":
        records = _split_excel_kfold(records, seed=seed, n_folds=n_folds, fold_index=fold_index, val_ratio=val_ratio)
    elif split_strategy == "random":
        records = _split_by_label(records, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}")

    data_text = str(data)
    root = Path(data_text.split(",")[0]).parent if "," in data_text else Path(data_text)
    return records, skipped, root


def discover_text_labels(root, splits=SPLITS):
    root = Path(root)
    labels = set()
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        labels.update(path.name for path in split_dir.iterdir() if path.is_dir())
        labels.update(path.stem for path in split_dir.glob("*.csv") if not path.name.startswith("._"))
    return sorted(labels)


def _join_csv_row(row):
    parts = []
    for value in row.values():
        value = (value or "").strip()
        if value:
            parts.append(value)
    return "\n".join(parts).strip()


def _collect_split_label_csv(root, splits, labels, label_to_id):
    records, skipped = [], []
    for split in splits:
        for label in labels:
            csv_path = root / split / f"{label}.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for index, row in enumerate(reader, start=1):
                    text = _join_csv_row(row)
                    if not text:
                        skipped.append(f"{csv_path}:{index}")
                        continue
                    records.append(
                        {
                            "id": f"{split}/{label}/{csv_path.stem}_{index:06d}",
                            "split": split,
                            "label": label_to_id[label],
                            "label_name": label,
                            "text_path": str(csv_path),
                            "text": text,
                        }
                    )
    return records, skipped


def collect_large_text(root, splits=SPLITS, labels=None, label_to_id=None):
    root = Path(root)
    labels = list(labels) if labels is not None else discover_text_labels(root, splits)
    if not labels:
        labels = list(LABELS)
    if label_to_id is None:
        label_to_id, _ = make_label_maps(labels)

    records, skipped = [], []
    for split in splits:
        for label in labels:
            label_dir = root / split / label
            if not label_dir.exists():
                continue
            for path in sorted(label_dir.glob("*.txt")):
                if path.name.startswith("._"):
                    skipped.append(str(path))
                    continue
                text = read_text(path).strip()
                if not text:
                    skipped.append(str(path))
                    continue
                records.append(
                    {
                        "id": f"{split}/{label}/{path.stem}",
                        "split": split,
                        "label": label_to_id[label],
                        "label_name": label,
                        "text_path": str(path),
                        "text": text,
                    }
                )
    if records:
        return records, skipped, root

    csv_records, csv_skipped = _collect_split_label_csv(root, splits, labels, label_to_id)
    records.extend(csv_records)
    skipped.extend(csv_skipped)
    return records, skipped, root


def collect_small_text(root, splits=SPLITS, labels=None, label_to_id=None):
    return collect_large_text(root, splits=splits, labels=labels, label_to_id=label_to_id)


def resolve_image_root(root):
    root = Path(root)
    if (root / "train").exists():
        return root
    if (root / "lan1-full" / "train").exists():
        return root / "lan1-full"
    raise FileNotFoundError(f"Cannot find split folders under {root}.")


def collect_paired_text(image_root, splits=SPLITS):
    image_root = resolve_image_root(image_root)
    records, missing = [], []
    for split in splits:
        for label in LABELS:
            label_dir = image_root / split / label
            if not label_dir.exists():
                continue
            for patient_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
                path = patient_dir / f"{patient_dir.name}.txt"
                pid = f"{split}/{label}/{patient_dir.name}"
                if not path.exists():
                    missing.append(pid)
                    continue
                text = read_text(path).strip()
                if not text:
                    missing.append(pid)
                    continue
                records.append(
                    {
                        "id": pid,
                        "split": split,
                        "label": LABEL_TO_ID[label],
                        "label_name": label,
                        "text_path": str(path),
                        "text": text,
                    }
                )
    return records, missing, image_root


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
    optional_fields = ["source_file", "row_index", "binary_label_name", "multiclass_label_name", "fold"]
    fieldnames = ["id", "split", "label", "label_name", "text_path"]
    fieldnames.extend(field for field in optional_fields if any(field in row for row in records))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
