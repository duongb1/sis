import csv

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from .common import ID_TO_LABEL, round_float


def _safe_auc(labels, probs, num_classes):
    try:
        if num_classes == 2:
            scores = probs[:, 1] if probs.ndim == 2 else probs
            return float(roc_auc_score(labels, scores))
        return float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def _binary_one_vs_rest_metrics(labels, probs, preds, positive_index, positive_label):
    true_binary = (labels == positive_index).astype(np.int64)
    pred_binary = (preds == positive_index).astype(np.int64)
    if probs.ndim == 2:
        positive_probs = probs[:, positive_index]
    else:
        positive_probs = probs
    tn, fp, fn, tp = confusion_matrix(true_binary, pred_binary, labels=[0, 1]).ravel()
    return {
        "positive_label": positive_label,
        "negative_label": f"NOT_{positive_label}",
        "accuracy": float(accuracy_score(true_binary, pred_binary)),
        "f1": float(f1_score(true_binary, pred_binary, zero_division=0)),
        "auc": _safe_auc(true_binary, positive_probs, 2),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "num_positive": int(true_binary.sum()),
        "num_negative": int((true_binary == 0).sum()),
    }


def cls_metrics(
    labels,
    probs,
    preds,
    loss=None,
    id_name="num_samples",
    threshold=0.5,
    label_names=None,
    binary_positive_label=None,
):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)
    preds = np.asarray(preds, dtype=np.int64)
    if probs.ndim == 1:
        num_classes = 2
    else:
        num_classes = probs.shape[1]
    label_ids = list(range(num_classes))
    label_names = list(label_names) if label_names is not None else [ID_TO_LABEL.get(i, str(i)) for i in label_ids]
    cm = confusion_matrix(labels, preds, labels=label_ids).tolist()
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "auc": _safe_auc(labels, probs, num_classes),
        "confusion_matrix": cm,
        id_name: int(len(labels)),
        "num_classes": int(num_classes),
        "class_names": label_names,
        "class_counts": {label_names[i]: int((labels == i).sum()) for i in label_ids},
        "threshold": threshold,
    }
    if num_classes == 2:
        tn, fp = cm[0]
        fn, tp = cm[1]
        metrics.update(
            {
                "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
                "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
            }
        )
    if binary_positive_label and binary_positive_label in label_names:
        positive_index = label_names.index(binary_positive_label)
        metrics["binary_i63"] = _binary_one_vs_rest_metrics(labels, probs, preds, positive_index, binary_positive_label)
    if loss is not None:
        metrics = {"loss": float(loss), **metrics}
    return metrics


def save_preds(path, ids, labels, probs, preds, id_field, label_names=None, binary_positive_label=None):
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim == 1:
        probs = np.stack([1.0 - probs, probs], axis=1)
    num_classes = probs.shape[1]
    label_names = list(label_names) if label_names is not None else [ID_TO_LABEL.get(i, str(i)) for i in range(num_classes)]
    prob_fields = [f"prob_{label}" for label in label_names]
    binary_fields = []
    positive_index = None
    if binary_positive_label and binary_positive_label in label_names:
        positive_index = label_names.index(binary_positive_label)
        binary_fields = ["true_binary_i63", "pred_binary_i63", "prob_binary_i63"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[id_field, "true_label", "true_name", "pred_label", "pred_name", *binary_fields, *prob_fields],
        )
        writer.writeheader()
        for item_id, label, prob_row, pred in zip(ids, labels, probs, preds):
            row = {
                id_field: item_id,
                "true_label": int(label),
                "true_name": label_names[int(label)] if int(label) < len(label_names) else str(label),
                "pred_label": int(pred),
                "pred_name": label_names[int(pred)] if int(pred) < len(label_names) else str(pred),
            }
            for field, prob in zip(prob_fields, prob_row):
                row[field] = round_float(prob)
            if positive_index is not None:
                row["true_binary_i63"] = int(int(label) == positive_index)
                row["pred_binary_i63"] = int(int(pred) == positive_index)
                row["prob_binary_i63"] = round_float(prob_row[positive_index])
            writer.writerow(row)


def format_metrics_summary(name, metrics):
    parts = [
        f"{name}:",
        f"loss={metrics.get('loss', float('nan'))}",
        f"acc={metrics.get('accuracy', float('nan'))}",
        f"f1_macro={metrics.get('f1_macro', metrics.get('f1', float('nan')))}",
        f"f1_weighted={metrics.get('f1_weighted', float('nan'))}",
        f"auc={metrics.get('auc', float('nan'))}",
    ]
    lines = [" ".join(str(part) for part in parts)]
    binary = metrics.get("binary_i63")
    if binary:
        lines.append(
            "binary_i63: "
            f"acc={binary.get('accuracy')} "
            f"f1={binary.get('f1')} "
            f"auc={binary.get('auc')} "
            f"sens={binary.get('sensitivity')} "
            f"spec={binary.get('specificity')} "
            f"cm={binary.get('confusion_matrix')}"
        )
    return "\n".join(lines)
