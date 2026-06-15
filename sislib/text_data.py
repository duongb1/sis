import csv
import random
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from .common import LABELS, LABEL_TO_ID, SPLITS, read_text
from .data.labels import EXCEL_MULTICLASS_LABELS, EXCEL_TEXT_COLUMNS, normalize_multiclass_label
from .data.splits import assign_kfold_splits


from .data.labels import EXCEL_MULTICLASS_LABEL_MAP  # Backward-compatible re-export.


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


def is_processed_csv_data(data):
    paths = _processed_csv_paths(data)
    if not paths:
        return False
    for path in paths:
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
        if {"Input_Text", "Label"}.issubset(fields):
            return True
    return False


def _processed_csv_paths(data):
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


def _processed_label_name(value):
    label = _clean_cell(value).lower()
    if label in {"1", "1.0", "co", "yes", "true", "i63", "i63_infarction"}:
        return "co"
    if label in {"0", "0.0", "khong", "không", "no", "false", "non_i63"}:
        return "khong"
    raise ValueError(f"Unsupported processed CSV binary label: {value!r}")


def _label_from_excel_filename(path):
    tokens = re.split(r"[^a-z0-9]+", path.stem.lower())
    if "khong" in tokens:
        return "khong"
    if "co" in tokens:
        return "co"
    raise ValueError(f"Cannot infer binary label from Excel filename: {path.name}")


def _excel_multiclass_label(value):
    label = _clean_cell(value)
    return normalize_multiclass_label(label)


def _join_excel_row(row):
    parts = []
    for column in EXCEL_TEXT_COLUMNS:
        if column not in row:
            continue
        value = _clean_cell(row[column])
        if value:
            parts.append(value)
    return "\n".join(parts).strip()


def _excel_row_fields(row):
    return {column: _clean_cell(row.get(column)) for column in EXCEL_TEXT_COLUMNS}


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


def _split_excel_kfold(records, seed=42, n_folds=5, fold_index=0, val_ratio=0.1, split_label="target"):
    return assign_kfold_splits(records, seed=seed, n_folds=n_folds, fold_index=fold_index, val_ratio=val_ratio, split_label=split_label)


def discover_excel_labels(data, task="multiclass"):
    if task == "binary":
        found = {_label_from_excel_filename(path) for path in _excel_paths(data)}
        return [label for label in sorted(found, key=lambda item: LABEL_TO_ID[item])]
    if task == "multitask":
        return ["non_i63", "I63_INFARCTION"]
    labels = set()
    for path in _excel_paths(data):
        df = _read_excel(path)
        if "LABEL" not in df.columns:
            raise ValueError(f"Missing LABEL column in {path}")
        labels.update(_excel_multiclass_label(value) for value in df["LABEL"].tolist())
    labels = {label for label in labels if label}
    ordered = [label for label in EXCEL_MULTICLASS_LABELS if label in labels]
    ordered.extend(sorted(labels - set(EXCEL_MULTICLASS_LABELS)))
    return ordered


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
    split_label="target",
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
            raw_multiclass_label = _clean_cell(row_dict.get("LABEL"))
            multiclass_label = _excel_multiclass_label(raw_multiclass_label)
            if task == "binary":
                label_name = binary_label_name
            elif task == "multitask":
                label_name = "I63_INFARCTION" if multiclass_label == "I63_INFARCTION" else "non_i63"
            else:
                label_name = multiclass_label
            if not label_name:
                skipped.append(f"{path}:{index + 2}:missing_label")
                continue
            if task == "multitask" and multiclass_label not in EXCEL_MULTICLASS_LABELS:
                skipped.append(f"{path}:{index + 2}:unknown_aux_label:{raw_multiclass_label}")
                continue
            if label_name not in label_to_id:
                skipped.append(f"{path}:{index + 2}:unknown_label:{label_name}")
                continue
            text = _join_excel_row(row_dict)
            if not text:
                skipped.append(f"{path}:{index + 2}:empty_text")
                continue
            record = {
                "id": f"{path.stem}_{index + 2:06d}",
                "split": "",
                "label": label_to_id[label_name],
                "label_name": label_name,
                "text_path": str(path),
                "text": text,
                "fields": _excel_row_fields(row_dict),
                "source_file": path.name,
                "row_index": index + 2,
                "binary_label_name": binary_label_name,
                "multiclass_label_name": multiclass_label,
                "raw_multiclass_label_name": raw_multiclass_label,
            }
            if multiclass_label in EXCEL_MULTICLASS_LABELS:
                record["multiclass_label"] = EXCEL_MULTICLASS_LABELS.index(multiclass_label)
            if task == "multitask":
                record["aux_label"] = EXCEL_MULTICLASS_LABELS.index(multiclass_label)
                record["aux_label_name"] = multiclass_label
            records.append(record)
    if split_strategy == "kfold":
        records = _split_excel_kfold(
            records,
            seed=seed,
            n_folds=n_folds,
            fold_index=fold_index,
            val_ratio=val_ratio,
            split_label=split_label,
        )
    elif split_strategy == "random":
        records = _split_by_label(records, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}")

    data_text = str(data)
    root = Path(data_text.split(",")[0]).parent if "," in data_text else Path(data_text)
    return records, skipped, root


def discover_processed_csv_labels(data):
    labels = set()
    for path in _processed_csv_paths(data):
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not {"Input_Text", "Label"}.issubset(set(reader.fieldnames or [])):
                continue
            for row in reader:
                labels.add(_processed_label_name(row.get("Label")))
    return [label for label in ["khong", "co"] if label in labels]


def collect_processed_csv_text(
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
    paths = _processed_csv_paths(data)
    if not paths:
        raise FileNotFoundError(f"Cannot find processed CSV files under {data}.")
    labels = list(labels) if labels is not None else discover_processed_csv_labels(data)
    if not labels:
        labels = ["khong", "co"]
    if label_to_id is None:
        label_to_id, _ = make_label_maps(labels)

    records, skipped = [], []
    for path in paths:
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            missing_columns = [column for column in ["Input_Text", "Label"] if column not in (reader.fieldnames or [])]
            if missing_columns:
                raise ValueError(f"{path} is missing required columns: {', '.join(missing_columns)}")
            for index, row in enumerate(reader, start=2):
                try:
                    label_name = _processed_label_name(row.get("Label"))
                except ValueError as exc:
                    skipped.append(f"{path}:{index}:{exc}")
                    continue
                if label_name not in label_to_id:
                    skipped.append(f"{path}:{index}:unknown_label:{label_name}")
                    continue
                text = _clean_cell(row.get("Input_Text"))
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
        records = _split_excel_kfold(
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
        if "aux_label" in row and row["aux_label"] != "":
            item["aux_labels"] = torch.tensor(row["aux_label"], dtype=torch.long)
        if "multiclass_label" in row and row["multiclass_label"] != "":
            item["multiclass_labels"] = torch.tensor(row["multiclass_label"], dtype=torch.long)
        item["id"] = row["id"]
        return item


def save_records(path, records):
    optional_fields = ["source_file", "row_index", "binary_label_name", "multiclass_label", "multiclass_label_name", "raw_multiclass_label_name", "aux_label", "aux_label_name", "fold"]
    fieldnames = ["id", "split", "label", "label_name", "text_path"]
    fieldnames.extend(field for field in optional_fields if any(field in row for row in records))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
