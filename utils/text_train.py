import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .common import batch_to_device, unwrap
from .metrics import cls_metrics


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_states, attention_mask):
        scores = self.proj(hidden_states).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class GatedClsAttnFusion(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.attn_pool = AttentionPooling(hidden_size, dropout=dropout)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, attention_mask):
        h_cls = hidden_states[:, 0, :]
        h_attn, attn_weights = self.attn_pool(hidden_states, attention_mask)
        fusion_features = torch.cat(
            [
                h_cls,
                h_attn,
                torch.abs(h_cls - h_attn),
                h_cls * h_attn,
            ],
            dim=-1,
        )
        gate = self.gate(fusion_features)
        h_fused = gate * h_cls + (1.0 - gate) * h_attn
        return self.norm(h_fused), attn_weights


class PhoBERTClassifier(nn.Module):
    base_model_prefix = "encoder"

    def __init__(self, model_name="vinai/phobert-base", num_labels=2, dropout=0.1, pooling="attention"):
        super().__init__()
        from transformers import AutoModel

        if pooling not in {"cls", "attention", "gated"}:
            raise ValueError(f"Unsupported pooling method: {pooling}")
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.pooling = pooling
        self.attn_pool = AttentionPooling(hidden_size, dropout=dropout) if pooling == "attention" else None
        self.gated_fusion = GatedClsAttnFusion(hidden_size, dropout=dropout) if pooling == "gated" else None
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.config = self.encoder.config
        self.config.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**model_inputs)
        hidden = outputs.last_hidden_state
        if self.pooling == "attention":
            pooled, attn_weights = self.attn_pool(hidden, attention_mask)
        elif self.pooling == "gated":
            pooled, attn_weights = self.gated_fusion(hidden, attention_mask)
        else:
            pooled, attn_weights = hidden[:, 0], None
        features = pooled
        logits = self.classifier(self.dropout(features))
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {
            "loss": loss,
            "logits": logits,
            "features": features,
            "attn_weights": attn_weights,
        }


def _output_loss(outputs):
    if isinstance(outputs, dict):
        return outputs.get("loss")
    return outputs.loss


def _output_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    return outputs.logits


def hf_loss(outputs, labels):
    loss = _output_loss(outputs)
    logits = _output_logits(outputs)
    loss = loss if loss is not None else F.cross_entropy(logits, labels)
    return loss.mean()


def ce_epoch(model, loader, optimizer, scheduler, scaler, device, accum=1):
    model.train()
    total_loss, total_count = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="Training", leave=False), start=1):
        inputs, _ = batch_to_device(batch, device, "id")
        inputs.pop("sample_weight", None)
        labels = inputs["labels"]
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(**inputs)
            loss = hf_loss(outputs, labels)
            scaled_loss = loss / accum
        scaler.scale(scaled_loss).backward()
        if step % accum == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        bs = labels.size(0)
        total_loss += loss.item() * bs
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
        outputs = unwrap(model)(**inputs)
        loss = hf_loss(outputs, labels)
        probs = torch.softmax(_output_logits(outputs), dim=-1)
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
    result = (
        cls_metrics(
            labels,
            probs,
            preds,
            loss=loss,
            threshold=threshold,
            label_names=label_names,
            binary_positive_label=binary_positive_label,
        ),
        ids,
        labels,
        probs,
        preds,
    )
    return result
