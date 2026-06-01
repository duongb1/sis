import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .common import batch_to_device, unwrap
from .metrics import cls_metrics


class PhoBERTMultiTask(nn.Module):
    base_model_prefix = "encoder"

    def __init__(self, model_name="vinai/phobert-base", dropout=0.1):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.binary_head = nn.Linear(hidden_size, 2)
        self.aux_head = nn.Linear(hidden_size, 3)
        self.config = self.encoder.config

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, aux_labels=None, lambda_aux=0.5, **kwargs):
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**model_inputs)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        binary_logits = self.binary_head(pooled)
        aux_logits = self.aux_head(pooled)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(binary_logits, labels)
            if aux_labels is not None:
                loss = loss + float(lambda_aux) * F.cross_entropy(aux_logits, aux_labels)
        return {
            "loss": loss,
            "logits": binary_logits,
            "binary_logits": binary_logits,
            "aux_logits": aux_logits,
        }


def hf_loss(outputs, labels):
    loss = outputs.loss if outputs.loss is not None else F.cross_entropy(outputs.logits, labels)
    return loss.mean()


def ce_epoch(model, loader, optimizer, scheduler, scaler, device, accum=1):
    model.train()
    total_loss, total_count = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="Training", leave=False), start=1):
        inputs, _ = batch_to_device(batch, device, "id")
        labels = inputs["labels"]
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss = hf_loss(model(**inputs), labels)
            loss = loss / accum
        scaler.scale(loss).backward()
        if step % accum == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        bs = labels.size(0)
        total_loss += loss.item() * accum * bs
        total_count += bs
    return float(total_loss / max(total_count, 1))


def multitask_epoch(model, loader, optimizer, scheduler, scaler, device, accum=1, lambda_aux=0.5):
    model.train()
    total_loss, total_count = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="Training", leave=False), start=1):
        inputs, _ = batch_to_device(batch, device, "id")
        inputs.pop("sample_weight", None)
        labels = inputs["labels"]
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss = model(**inputs, lambda_aux=lambda_aux)["loss"]
            loss = loss / accum
        scaler.scale(loss).backward()
        if step % accum == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        bs = labels.size(0)
        total_loss += loss.item() * accum * bs
        total_count += bs
    return float(total_loss / max(total_count, 1))


@torch.no_grad()
def eval_text(model, loader, device, threshold=0.5, desc="Evaluating", label_names=None, binary_positive_label=None):
    model.eval()
    total_loss, total_count = 0.0, 0
    ids, labels_all, probs_all = [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        inputs, batch_ids = batch_to_device(batch, device, "id")
        inputs.pop("sample_weight", None)
        labels = inputs["labels"]
        outputs = model(**inputs)
        loss = hf_loss(outputs, labels)
        probs = torch.softmax(outputs.logits, dim=-1)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_count += bs
        ids.extend(batch_ids)
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        probs_all.extend(probs.detach().cpu().numpy().tolist())
    labels = np.array(labels_all, dtype=np.int64)
    probs = np.array(probs_all, dtype=np.float32)
    if probs.ndim == 2 and probs.shape[1] == 2:
        preds = (probs[:, 1] >= threshold).astype(np.int64)
    else:
        preds = probs.argmax(axis=1).astype(np.int64)
    loss = total_loss / max(total_count, 1)
    return cls_metrics(
        labels,
        probs,
        preds,
        loss=loss,
        threshold=threshold,
        label_names=label_names,
        binary_positive_label=binary_positive_label,
    ), ids, labels, probs, preds


@torch.no_grad()
def eval_multitask(model, loader, device, threshold=0.5, desc="Evaluating", binary_label_names=None, aux_label_names=None):
    model.eval()
    total_loss, total_count = 0.0, 0
    ids, labels_all, aux_labels_all, binary_probs_all, aux_probs_all = [], [], [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        inputs, batch_ids = batch_to_device(batch, device, "id")
        inputs.pop("sample_weight", None)
        labels = inputs["labels"]
        aux_labels = inputs["aux_labels"]
        outputs = model(**inputs)
        loss = outputs["loss"]
        binary_probs = torch.softmax(outputs["binary_logits"], dim=-1)
        aux_probs = torch.softmax(outputs["aux_logits"], dim=-1)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_count += bs
        ids.extend(batch_ids)
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        aux_labels_all.extend(aux_labels.detach().cpu().numpy().tolist())
        binary_probs_all.extend(binary_probs.detach().cpu().numpy().tolist())
        aux_probs_all.extend(aux_probs.detach().cpu().numpy().tolist())
    labels = np.array(labels_all, dtype=np.int64)
    aux_labels = np.array(aux_labels_all, dtype=np.int64)
    binary_probs = np.array(binary_probs_all, dtype=np.float32)
    aux_probs = np.array(aux_probs_all, dtype=np.float32)
    binary_preds = (binary_probs[:, 1] >= threshold).astype(np.int64)
    aux_preds = aux_probs.argmax(axis=1).astype(np.int64)
    loss = total_loss / max(total_count, 1)
    binary_label_names = binary_label_names or ["non_i63", "I63_INFARCTION"]
    aux_label_names = aux_label_names or ["I63_INFARCTION", "OTHER_STROKE_LIKE", "DISTANT_OTHER"]
    binary_metrics = cls_metrics(
        labels,
        binary_probs,
        binary_preds,
        loss=loss,
        threshold=threshold,
        label_names=binary_label_names,
        binary_positive_label="I63_INFARCTION",
    )
    aux_metrics = cls_metrics(
        aux_labels,
        aux_probs,
        aux_preds,
        loss=loss,
        threshold=threshold,
        label_names=aux_label_names,
        binary_positive_label="I63_INFARCTION",
    )
    return (
        {"primary_binary": binary_metrics, "aux_3class": aux_metrics, "loss": loss},
        ids,
        labels,
        binary_probs,
        binary_preds,
        aux_labels,
        aux_probs,
        aux_preds,
    )
