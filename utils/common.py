import random
import os
import logging
import warnings
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LABEL_TO_ID = {"khong": 0, "co": 1}
ID_TO_LABEL = {0: "khong", 1: "co"}
SPLITS = ["train", "val", "test"]
LABELS = ["co", "khong"]


def quiet_hf_logging():
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message="Was asked to gather along dimension 0, but all input tensors were scalars.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*You are sending unauthenticated requests to the HF Hub.*",
        category=UserWarning,
    )
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass


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


class AutocastDPWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        is_cuda = False
        if args:
            is_cuda = any(isinstance(x, torch.Tensor) and x.is_cuda for x in args)
        if not is_cuda and kwargs:
            is_cuda = any(isinstance(x, torch.Tensor) and x.is_cuda for x in kwargs.values())
        device_type = "cuda" if is_cuda else "cpu"
        with torch.amp.autocast(device_type, enabled=is_cuda):
            return self.module(*args, **kwargs)


def unwrap(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if isinstance(model, AutocastDPWrapper):
        model = model.module
    return model


def to_device(model, device, multi_gpu=True):
    model = model.to(device)
    if device.type == "cuda" and multi_gpu and torch.cuda.device_count() > 1:
        ids = list(range(torch.cuda.device_count()))
        print(f"Using DataParallel on GPU ids: {ids}")
        model = torch.nn.DataParallel(AutocastDPWrapper(model), device_ids=ids)
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
        elif isinstance(value, dict):
            out[key] = round_metrics(value, digits)
        elif isinstance(value, list):
            out[key] = [
                round_float(item, digits) if isinstance(item, (float, np.floating)) else item
                for item in value
            ]
        else:
            out[key] = value
    return out


def resolve_max_len(model, requested):
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is None:
        return requested

    base_model = getattr(model, getattr(model, "base_model_prefix", ""), None)
    embeddings = getattr(base_model, "embeddings", None) if base_model is not None else None
    position_embeddings = getattr(embeddings, "position_embeddings", None)
    if position_embeddings is None:
        return requested

    padding_idx = position_embeddings.padding_idx or 0
    required_positions = requested + padding_idx + 1
    if max_positions >= required_positions and position_embeddings.num_embeddings >= required_positions:
        return requested

    old_weight = position_embeddings.weight.data
    new_embeddings = nn.Embedding(required_positions, old_weight.shape[1], padding_idx=position_embeddings.padding_idx)
    new_embeddings.to(old_weight.device, dtype=old_weight.dtype)
    rows_to_copy = min(old_weight.shape[0], required_positions)
    new_embeddings.weight.data[:rows_to_copy] = old_weight[:rows_to_copy]
    if required_positions > rows_to_copy:
        new_embeddings.weight.data[rows_to_copy:] = old_weight[-1].unsqueeze(0).repeat(required_positions - rows_to_copy, 1)
    embeddings.position_embeddings = new_embeddings
    embeddings.register_buffer("position_ids", torch.arange(required_positions).expand((1, -1)), persistent=False)
    if hasattr(embeddings, "token_type_ids"):
        embeddings.register_buffer("token_type_ids", torch.zeros((1, required_positions), dtype=torch.long), persistent=False)
    model.config.max_position_embeddings = required_positions
    if hasattr(base_model, "config"):
        base_model.config.max_position_embeddings = required_positions
    return requested


def split_records(records, split):
    return [r for r in records if r["split"] == split]


def batch_to_device(batch, device, id_key):
    ids = batch.pop(id_key)
    inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    return inputs, ids


def build_python_env(root):
    env = os.environ.copy()
    root = Path(root)
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(root) if not pythonpath else os.pathsep.join([str(root), pythonpath])
    return env


def run_stage(name, cmd, done_path, force=False, dry_run=False, cwd=None):
    cwd = Path(cwd or Path.cwd())
    done_path = Path(done_path)
    print("\n" + "=" * 80)
    print(name)
    print(" ".join(str(item) for item in cmd), flush=True)
    if done_path.exists() and not force:
        print(f"Skip: found {done_path}")
        return
    if dry_run:
        return
    subprocess.run([str(item) for item in cmd], check=True, cwd=cwd, env=build_python_env(cwd))
