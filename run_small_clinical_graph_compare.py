import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sislib.clinical_concepts import (
    CONCEPT_NAMES,
    TEXT_FIELDS_ALL,
    TEXT_FIELDS_CHIEF_EXAM,
    build_text,
    extract_concepts_by_field,
)
from sislib.clinical_graph import NODE_TYPES, build_clinical_graph, node_name_vocab
from sislib.clinical_graph_model import ClinicalGraphOnlyClassifier, PhoBERTClinicalGraphFusion
from sislib.common import get_device, quiet_hf_logging, round_float, seed_all
from sislib.text_data import EXCEL_MULTICLASS_LABEL_MAP
from sislib.text_train import PhoBERTClassifier

quiet_hf_logging()


VALID_MULTICLASS_LABELS = {"DISTANT_OTHER", "I63_INFARCTION", "OTHER_STROKE_LIKE"}
MULTICLASS_LABELS = ["DISTANT_OTHER", "I63_INFARCTION", "OTHER_STROKE_LIKE"]
BINARY_MAPPING = {"I63_INFARCTION": 1, "OTHER_STROKE_LIKE": 0, "DISTANT_OTHER": 0}
MODEL_NAMES = [
    "small_binary_attnpool",
    "small_concept_graph_only",
    "small_phobert_attnpool_clinical_graph_fusion",
]
REFERENCE_BASELINE = {
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
    parser = argparse.ArgumentParser(description="Run small PhoBERT + clinical concept graph 5-fold comparison.")
    parser.add_argument("--excel-root", default="/kaggle/input/datasets/duongbui/siscth")
    parser.add_argument("--output-dir", default="/kaggle/working/sis_excel_5fold_clinical_graph_mcstrat")
    parser.add_argument("--model", default="vinai/phobert-base")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--pooling", choices=["cls", "attention"], default="attention")
    parser.add_argument("--input-mode", choices=["concat"], default="concat")
    parser.add_argument("--max-len-per-field", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--excel-split-label", choices=["multiclass"], default="multiclass")
    parser.add_argument("--graph-hidden-dim", type=int, default=64)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-dropout", type=float, default=0.2)
    parser.add_argument("--graph-heads", type=int, default=2)
    parser.add_argument("--graph-pooling", choices=["patient", "mean"], default="patient")
    parser.add_argument("--graph-conv", choices=["gat", "gcn"], default="gat")
    parser.add_argument("--concept-fields", choices=["all_fields", "chief_exam"], default="all_fields")
    parser.add_argument("--only", default=",".join(MODEL_NAMES), help="Comma-separated model names to run.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-mgpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_thresholds(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def selected_fields(value):
    return TEXT_FIELDS_ALL if value == "all_fields" else TEXT_FIELDS_CHIEF_EXAM


def excel_paths(excel_root):
    root = Path(excel_root)
    return [root / "700_co_label.xlsx", root / "700_khong_label.xlsx"]


def load_excel(excel_root):
    frames = []
    for path in excel_paths(excel_root):
        if not path.exists():
            raise FileNotFoundError(f"Missing required Excel file: {path}")
        df = pd.read_excel(path)
        missing = [column for column in [*TEXT_FIELDS_ALL, "LABEL"] if column not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        df["source_file"] = path.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    raw_bad = set(df["LABEL"].dropna().astype(str).unique()) - set(EXCEL_MULTICLASS_LABEL_MAP)
    if raw_bad:
        raise ValueError(f"Unexpected raw LABEL values before 3-class mapping: {raw_bad}")
    df["raw_LABEL"] = df["LABEL"]
    df["LABEL"] = df["LABEL"].map(EXCEL_MULTICLASS_LABEL_MAP)
    bad_labels = set(df["LABEL"].dropna().unique()) - VALID_MULTICLASS_LABELS
    if bad_labels:
        raise ValueError(f"Unexpected LABEL values: {bad_labels}")
    df["binary_label"] = df["LABEL"].map(BINARY_MAPPING).astype(int)
    print("Multiclass distribution:")
    print(df["LABEL"].value_counts())
    return df.reset_index(drop=True)


def make_splits(df, args):
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    val_size_relative = args.val_ratio / (1.0 - args.test_ratio)
    for fold, (trainval_idx, test_idx) in enumerate(skf.split(df, df["LABEL"])):
        trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        train_df, val_df = train_test_split(
            trainval_df,
            test_size=val_size_relative,
            random_state=args.seed + fold,
            stratify=trainval_df["LABEL"],
        )
        yield fold, train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


class ClinicalGraphTextDataset(Dataset):
    def __init__(self, df, tokenizer, fields, max_len, node_name_to_id):
        self.rows = df.to_dict("records")
        self.tokenizer = tokenizer
        self.fields = fields
        self.max_len = max_len
        self.node_name_to_id = node_name_to_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        label = int(row["binary_label"])
        graph = build_clinical_graph(row, self.fields, label, self.node_name_to_id)
        item = {
            "labels": torch.tensor(label, dtype=torch.float32),
            "labels_long": torch.tensor(label, dtype=torch.long),
            "multiclass_label": row["LABEL"],
            "graph": graph,
            "detected_concepts": graph.detected_concepts,
        }
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                build_text(row, self.fields),
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
            item.update({key: value.squeeze(0) for key, value in encoded.items()})
        return item


def collate_graph_text(batch):
    try:
        from torch_geometric.data import Batch
    except ImportError as exc:
        raise ImportError("PyTorch Geometric is required. Install it with: pip install torch_geometric") from exc
    result = {
        "labels": torch.stack([item["labels"] for item in batch]),
        "labels_long": torch.stack([item["labels_long"] for item in batch]),
        "graphs": Batch.from_data_list([item["graph"] for item in batch]),
        "detected_concepts": [item["detected_concepts"] for item in batch],
    }
    if "input_ids" in batch[0]:
        result["input_ids"] = torch.stack([item["input_ids"] for item in batch])
        result["attention_mask"] = torch.stack([item["attention_mask"] for item in batch])
        if "token_type_ids" in batch[0]:
            result["token_type_ids"] = torch.stack([item["token_type_ids"] for item in batch])
    return result


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        if key == "graphs":
            moved[key] = value.to(device)
        elif torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def binary_metrics(y_true, y_prob, threshold):
    y_pred = (np.asarray(y_prob) >= threshold).astype(np.int64)
    y_true = np.asarray(y_true).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
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


def logits_for_batch(model_name, model, batch, debug=False):
    if model_name == "small_binary_attnpool":
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels_long"])
        logits_2 = outputs["logits"]
        logits = logits_2[:, 1] - logits_2[:, 0]
        if debug:
            return {"logits": logits, "z_text": outputs["features"]}
        return logits
    if model_name == "small_concept_graph_only":
        return model(batch["graphs"], return_debug=debug)
    return model(batch["input_ids"], batch["attention_mask"], batch["graphs"], return_debug=debug)


def train_one_epoch(model_name, model, loader, optimizer, scheduler, device, accum):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_count = 0.0, 0
    for step, raw_batch in enumerate(tqdm(loader, desc="Training", leave=False), start=1):
        batch = move_batch(raw_batch, device)
        logits = logits_for_batch(model_name, model, batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
        (loss / accum).backward()
        if step % accum == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * batch["labels"].size(0)
        total_count += batch["labels"].size(0)
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model_name, model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    for raw_batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = move_batch(raw_batch, device)
        logits = logits_for_batch(model_name, model, batch)
        probs = torch.sigmoid(logits)
        y_true.extend(batch["labels"].detach().cpu().numpy().astype(int).tolist())
        y_prob.extend(probs.detach().cpu().numpy().tolist())
    return np.asarray(y_true), np.asarray(y_prob)


def debug_first_batch(model_name, model, loader, device):
    raw_batch = next(iter(loader))
    batch = move_batch(raw_batch, device)
    debug_out = logits_for_batch(model_name, model, batch, debug=True)
    logits = debug_out["logits"]
    graphs = batch["graphs"]
    patient_nodes = int((graphs.node_type_id == NODE_TYPES["patient"]).sum().item())
    batch_size = int(batch["labels"].size(0))
    type_counts = {}
    for name, type_id in NODE_TYPES.items():
        type_counts[name] = int((graphs.node_type_id == type_id).sum().item())
    print("[clinical graph debug]")
    print(f"batch_size={batch_size}")
    print(f"num_graphs={graphs.num_graphs}")
    print(f"num_nodes={graphs.num_nodes}")
    print(f"num_edges={graphs.edge_index.size(1)}")
    print(f"patient_nodes={patient_nodes}")
    print(f"node_type_counts={type_counts}")
    print(f"detected_concepts_sample0={batch['detected_concepts'][0]}")
    if "z_text" in debug_out:
        print(f"z_text.shape={list(debug_out['z_text'].shape)}")
    if "z_graph" in debug_out:
        print(f"z_graph.shape={list(debug_out['z_graph'].shape)}")
    print(f"logits.shape={list(logits.shape)}")
    if patient_nodes != batch_size:
        raise RuntimeError(f"patient_nodes != batch_size: {patient_nodes} != {batch_size}")
    if "z_text" in debug_out and "z_graph" in debug_out and debug_out["z_text"].shape[0] != debug_out["z_graph"].shape[0]:
        raise RuntimeError("z_text.shape[0] != z_graph.shape[0]")


def threshold_sweep(y_true, y_prob, thresholds):
    return {f"{threshold:g}": binary_metrics(y_true, y_prob, threshold) for threshold in thresholds}


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, ensure_ascii=False, indent=2, allow_nan=False)


def concept_stats(df, fields):
    stats = {label: Counter() for label in MULTICLASS_LABELS}
    for row in df.to_dict("records"):
        concepts_by_field = extract_concepts_by_field(row, fields)
        concepts = set().union(*concepts_by_field.values()) if concepts_by_field else set()
        stats[row["LABEL"]].update(concepts)
    return {label: dict(counter.most_common()) for label, counter in stats.items()}


def print_top_concepts(stats):
    print("Top concepts by class:")
    for label in MULTICLASS_LABELS:
        top = list(stats.get(label, {}))[:5]
        print(f"{label}: {', '.join(top) if top else '(none)'}")


def build_model(model_name, args, node_name_to_id):
    if model_name == "small_binary_attnpool":
        return PhoBERTClassifier(args.model, num_labels=2, pooling=args.pooling)
    if model_name == "small_concept_graph_only":
        return ClinicalGraphOnlyClassifier(
            len(node_name_to_id),
            graph_hidden_dim=args.graph_hidden_dim,
            graph_layers=args.graph_layers,
            graph_dropout=args.graph_dropout,
            graph_heads=args.graph_heads,
            graph_conv=args.graph_conv,
            graph_pooling=args.graph_pooling,
        )
    if model_name == "small_phobert_attnpool_clinical_graph_fusion":
        return PhoBERTClinicalGraphFusion(
            args.model,
            len(node_name_to_id),
            graph_hidden_dim=args.graph_hidden_dim,
            graph_layers=args.graph_layers,
            graph_dropout=args.graph_dropout,
            graph_heads=args.graph_heads,
            graph_conv=args.graph_conv,
            graph_pooling=args.graph_pooling,
            pooling=args.pooling,
        )
    raise ValueError(f"Unknown model: {model_name}")


def label_schema():
    return {
        "multiclass_labels": MULTICLASS_LABELS,
        "positive_label": "I63_INFARCTION",
        "binary_mapping": BINARY_MAPPING,
    }


def graph_metadata(args):
    return {
        "node_types": list(NODE_TYPES),
        "concepts": CONCEPT_NAMES,
        "graph_hidden_dim": args.graph_hidden_dim,
        "graph_layers": args.graph_layers,
        "graph_conv": args.graph_conv,
        "graph_pooling": args.graph_pooling,
    }


def run_model_fold(model_name, fold, train_df, val_df, test_df, fields, node_name_to_id, tokenizer, device, args, thresholds):
    out_dir = Path(args.output_dir) / model_name / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and best_path.exists() and not args.force:
        raise FileExistsError(f"Output exists, use --force to overwrite: {out_dir}")

    needs_text = model_name != "small_concept_graph_only"
    ds_tokenizer = tokenizer if needs_text else None
    train_loader = DataLoader(
        ClinicalGraphTextDataset(train_df, ds_tokenizer, fields, args.max_len, node_name_to_id),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_graph_text,
    )
    val_loader = DataLoader(
        ClinicalGraphTextDataset(val_df, ds_tokenizer, fields, args.max_len, node_name_to_id),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_graph_text,
    )
    test_loader = DataLoader(
        ClinicalGraphTextDataset(test_df, ds_tokenizer, fields, args.max_len, node_name_to_id),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_graph_text,
    )

    model = build_model(model_name, args, node_name_to_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = max(1, int(np.ceil(len(train_loader) / max(args.accum, 1))) * args.epochs)
    warmup_steps = int(total_steps * args.warmup)
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if fold == 0:
        debug_first_batch(model_name, model, train_loader, device)

    best_auc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model_name, model, train_loader, optimizer, scheduler, device, args.accum)
        val_y, val_prob = evaluate(model_name, model, val_loader, device)
        val_metrics = binary_metrics(val_y, val_prob, args.threshold)
        val_auc = val_metrics["auc"]
        score = val_auc if not math.isnan(val_auc) else val_metrics["f1"]
        print(f"{model_name} fold={fold} epoch={epoch:03d} train_loss={train_loss:.4f} val_auc={val_auc:.4f} val_f1={val_metrics['f1']:.4f}")
        if score > best_auc:
            best_auc = score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(best_state, best_path)
    model.load_state_dict(best_state)

    test_y, test_prob = evaluate(model_name, model, test_loader, device)
    test_metrics = binary_metrics(test_y, test_prob, args.threshold)
    sweep = threshold_sweep(test_y, test_prob, thresholds)
    stats_payload = {"train": concept_stats(train_df, fields), "test": concept_stats(test_df, fields)}
    save_json(out_dir / "concept_stats.json", stats_payload)
    print_top_concepts(stats_payload["train"])
    payload = {
        "model": model_name,
        "fold": fold,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "label_schema": label_schema(),
        "graph": graph_metadata(args),
        "fields": fields,
        "test": test_metrics,
        "threshold_sweep": {"test": sweep},
    }
    save_json(metrics_path, payload)
    return payload


def mean_std(values):
    arr = np.asarray([value for value in values if not math.isnan(float(value))], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def mean_std_text(values):
    mean, std = mean_std(values)
    return "nan" if math.isnan(mean) else f"{mean:.3f}±{std:.3f}"


def aggregate_cm(rows):
    tn = sum(row["test"]["tn"] for row in rows)
    fp = sum(row["test"]["fp"] for row in rows)
    fn = sum(row["test"]["fn"] for row in rows)
    tp = sum(row["test"]["tp"] for row in rows)
    return tn, fp, fn, tp


def print_summary(results):
    print("\n5-fold summary: clinical graph models\n")
    print(f"{'Model':52} {'Acc':9} {'F1':9} {'AUC':9} {'Sens':9} {'Spec':9} {'BalAcc':9} FP/FN")
    baseline_rows = results.get("small_binary_attnpool", [])
    baseline_mean = {}
    if baseline_rows:
        for key in ["f1", "auc", "sensitivity", "specificity", "balanced_accuracy"]:
            baseline_mean[key] = mean_std([row["test"][key] for row in baseline_rows])[0]
        _, baseline_fp, baseline_fn, _ = aggregate_cm(baseline_rows)
    else:
        baseline_fp, baseline_fn = REFERENCE_BASELINE["fp"], REFERENCE_BASELINE["fn"]
        baseline_mean = {key: REFERENCE_BASELINE[key] for key in ["f1", "auc", "sensitivity", "specificity", "balanced_accuracy"]}

    for model_name, rows in results.items():
        tn, fp, fn, tp = aggregate_cm(rows)
        print(
            f"{model_name:52} "
            f"{mean_std_text([row['test']['accuracy'] for row in rows]):9} "
            f"{mean_std_text([row['test']['f1'] for row in rows]):9} "
            f"{mean_std_text([row['test']['auc'] for row in rows]):9} "
            f"{mean_std_text([row['test']['sensitivity'] for row in rows]):9} "
            f"{mean_std_text([row['test']['specificity'] for row in rows]):9} "
            f"{mean_std_text([row['test']['balanced_accuracy'] for row in rows]):9} "
            f"{fp}/{fn}"
        )

    print("\nAggregate confusion counts:")
    for model_name, rows in results.items():
        tn, fp, fn, tp = aggregate_cm(rows)
        print(f"{model_name:52} TN={tn} FP={fp} FN={fn} TP={tp}")

    print("\nBest metrics:")
    flat = [(model_name, row) for model_name, rows in results.items() for row in rows]
    for display, key, selector in [
        ("best_f1", "f1", max),
        ("best_auc", "auc", max),
        ("best_sensitivity", "sensitivity", max),
        ("best_specificity", "specificity", max),
        ("best_balanced_accuracy", "balanced_accuracy", max),
        ("lowest_fn", "fn", min),
        ("lowest_fp", "fp", min),
    ]:
        candidates = [(name, row) for name, row in flat if not math.isnan(float(row["test"][key]))]
        name, row = selector(candidates, key=lambda item: item[1]["test"][key])
        print(f"{display}: {row['test'][key]} ({name}, fold {row['fold']})")

    print("\nDelta vs small_binary_attnpool:")
    for model_name, rows in results.items():
        if model_name == "small_binary_attnpool":
            continue
        tn, fp, fn, tp = aggregate_cm(rows)
        deltas = {}
        for key in ["f1", "auc", "sensitivity", "specificity", "balanced_accuracy"]:
            deltas[key] = mean_std([row["test"][key] for row in rows])[0] - baseline_mean[key]
        print(
            f"{model_name}: dF1={deltas['f1']:.3f} dAUC={deltas['auc']:.3f} "
            f"dSens={deltas['sensitivity']:.3f} dSpec={deltas['specificity']:.3f} "
            f"dBalAcc={deltas['balanced_accuracy']:.3f} dFP={fp - baseline_fp} dFN={fn - baseline_fn}"
        )


def main():
    args = parse_args()
    seed_all(args.seed)
    fields = selected_fields(args.concept_fields)
    selected = [item.strip() for item in args.only.split(",") if item.strip()]
    unknown = set(selected) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown model names in --only: {sorted(unknown)}")
    device = get_device(args.cpu)
    df = load_excel(args.excel_root)
    node_name_to_id = node_name_vocab(fields)
    thresholds = parse_thresholds(args.thresholds)

    tokenizer = None
    if any(name != "small_concept_graph_only" for name in selected):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = defaultdict(list)
    for fold, train_df, val_df, test_df in make_splits(df, args):
        print(f"\nFold {fold}: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
        for model_name in selected:
            print(f"\nRunning {model_name}")
            payload = run_model_fold(model_name, fold, train_df, val_df, test_df, fields, node_name_to_id, tokenizer, device, args, thresholds)
            results[model_name].append(payload)
    print_summary(results)


if __name__ == "__main__":
    main()
