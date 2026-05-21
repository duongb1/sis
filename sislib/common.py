import random
from pathlib import Path

import numpy as np
import torch


LABEL_TO_ID = {"khong": 0, "co": 1}
ID_TO_LABEL = {0: "khong", 1: "co"}
SPLITS = ["train", "val", "test"]
LABELS = ["co", "khong"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")

    supported_sms = getattr(torch.cuda, "get_arch_list", lambda: [])()
    for index in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(index)
        name = torch.cuda.get_device_name(index)
        sm = f"sm_{major}{minor}"
        if supported_sms and sm not in supported_sms:
            print(
                f"CUDA device {index} ({name}) has capability {sm}, but this PyTorch "
                f"build supports {supported_sms}. Falling back to CPU."
            )
            return torch.device("cpu")

    print(f"Using CUDA devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    return torch.device("cuda")


def unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def to_device(model, device, multi_gpu=True):
    model = model.to(device)
    if device.type == "cuda" and multi_gpu and torch.cuda.device_count() > 1:
        ids = list(range(torch.cuda.device_count()))
        print(f"Using DataParallel on GPU ids: {ids}")
        model = torch.nn.DataParallel(model, device_ids=ids)
    return model


def read_text(path):
    for enc in ("utf-8", "utf-8-sig", "cp1258", "latin-1"):
        try:
            return Path(path).read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def round_float(value, digits=3):
    return round(float(value), digits)


def round_metrics(metrics, digits=3):
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            out[key] = round_float(value, digits)
        else:
            out[key] = value
    return out


def resolve_max_len(model, requested):
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is None:
        return requested
    return min(requested, max_positions - 2)


def split_records(records, split):
    return [r for r in records if r["split"] == split]


def batch_to_device(batch, device, id_key):
    ids = batch.pop(id_key)
    inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    return inputs, ids
