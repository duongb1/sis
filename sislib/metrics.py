import csv

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .common import ID_TO_LABEL, round_float


def cls_metrics(labels, probs, preds, loss=None, id_name="num_samples", threshold=0.5):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)
    preds = np.asarray(preds, dtype=np.int64)
    tp = int(((labels == 1) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else float("nan"),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        id_name: int(len(labels)),
        "num_co": int((labels == 1).sum()),
        "num_khong": int((labels == 0).sum()),
        "threshold": threshold,
    }
    if loss is not None:
        metrics = {"loss": float(loss), **metrics}
    return metrics


def save_preds(path, ids, labels, probs, preds, id_field):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[id_field, "true_label", "true_name", "prob_co", "pred_label", "pred_name"],
        )
        writer.writeheader()
        for item_id, label, prob, pred in zip(ids, labels, probs, preds):
            writer.writerow(
                {
                    id_field: item_id,
                    "true_label": int(label),
                    "true_name": ID_TO_LABEL[int(label)],
                    "prob_co": round_float(prob),
                    "pred_label": int(pred),
                    "pred_name": ID_TO_LABEL[int(pred)],
                }
            )
