import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .mri import MRIDataset, resnet50_binary
from .common import round_metrics, to_device, unwrap


def clean_state_dict(state):
    if any(k.startswith("module.") for k in state):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def load_mri_teacher(path, device, multi_gpu):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model = resnet50_binary()
    model.load_state_dict(clean_state_dict(state))
    return to_device(model.eval(), device, multi_gpu)


@torch.no_grad()
def compute_mri_logits(text_records, mri_records, teacher, device, batch_size, workers):
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    loader = DataLoader(
        MRIDataset(mri_records, tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    logits_by_id = {}
    for images, _, ids in tqdm(loader, desc="Computing MRI teacher logits"):
        logits = teacher(images.to(device, non_blocking=True)).squeeze(1).detach().cpu().numpy()
        for logit, item_id in zip(logits, ids):
            logits_by_id.setdefault(item_id, []).append(float(logit))

    text_ids = {row["id"] for row in text_records}
    out = {}
    for item_id, values in logits_by_id.items():
        if item_id in text_ids:
            z = float(np.mean(values))
            out[item_id] = [-z / 2.0, z / 2.0]
    return out


@torch.no_grad()
def compute_mri_features(text_records, mri_records, teacher, device, batch_size, workers):
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    loader = DataLoader(
        MRIDataset(mri_records, tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    base = unwrap(teacher)
    features_by_id = {}
    for images, _, ids in tqdm(loader, desc="Computing MRI teacher features"):
        x = images.to(device, non_blocking=True)
        x = base.conv1(x)
        x = base.bn1(x)
        x = base.relu(x)
        x = base.maxpool(x)
        x = base.layer1(x)
        x = base.layer2(x)
        x = base.layer3(x)
        x = base.layer4(x)
        x = base.avgpool(x)
        feats = torch.flatten(x, 1).detach().cpu().numpy()
        for feat, item_id in zip(feats, ids):
            features_by_id.setdefault(item_id, []).append(feat.astype(np.float32))

    text_ids = {row["id"] for row in text_records}
    out = {}
    for item_id, values in features_by_id.items():
        if item_id in text_ids:
            out[item_id] = np.mean(np.stack(values, axis=0), axis=0).astype(np.float32).tolist()
    return out


def teacher_stats_for_records(records, teacher_logits, threshold):
    rows = [row for row in records if row["id"] in teacher_logits]
    labels = np.array([row["label"] for row in rows], dtype=np.int64)
    logits = np.array([teacher_logits[row["id"]] for row in rows], dtype=np.float32)
    if len(rows):
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits[:, 1] / exp_logits.sum(axis=1)
        preds = (probs >= threshold).astype(np.int64)
    else:
        probs = np.array([], dtype=np.float32)
        preds = np.array([], dtype=np.int64)
    tp = int(((labels == 1) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    return {
        "teacher_accuracy": float(accuracy_score(labels, preds)) if len(labels) else float("nan"),
        "teacher_f1": float(f1_score(labels, preds, zero_division=0)) if len(labels) else float("nan"),
        "teacher_auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else float("nan"),
        "teacher_sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "teacher_specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "teacher_num_patients": int(len(labels)),
        "teacher_correct_patients": int((preds == labels).sum()) if len(labels) else 0,
        "teacher_missing_patients": int(len(records) - len(rows)),
    }


def split_teacher_stats(train_rows, val_rows, test_rows, teacher_logits, threshold):
    return {
        "train": round_metrics(teacher_stats_for_records(train_rows, teacher_logits, threshold)),
        "val": round_metrics(teacher_stats_for_records(val_rows, teacher_logits, threshold)),
        "test": round_metrics(teacher_stats_for_records(test_rows, teacher_logits, threshold)),
    }


def shuffle_teacher_for_train(teacher_logits, train_rows, seed):
    rng = random.Random(seed)
    ids = [row["id"] for row in train_rows if row["id"] in teacher_logits]
    shuffled = ids[:]
    rng.shuffle(shuffled)
    out = dict(teacher_logits)
    for target_id, source_id in zip(ids, shuffled):
        out[target_id] = teacher_logits[source_id]
    return out
