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


class PhoBERTClassifier(nn.Module):
    base_model_prefix = "encoder"

    def __init__(self, model_name="vinai/phobert-base", num_labels=2, dropout=0.1, pooling="attention"):
        super().__init__()
        from transformers import AutoModel

        if pooling not in {"cls", "attention"}:
            raise ValueError(f"Unsupported pooling method: {pooling}")
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.pooling = pooling
        self.attn_pool = AttentionPooling(hidden_size, dropout=dropout) if pooling == "attention" else None
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
        else:
            pooled, attn_weights = hidden[:, 0], None
        logits = self.classifier(self.dropout(pooled))
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits, "attn_weights": attn_weights}


class FieldAwarePhoBERTClassifier(nn.Module):
    base_model_prefix = "encoder"

    def __init__(
        self,
        model_name="vinai/phobert-base",
        num_labels=2,
        num_fields=6,
        dropout=0.1,
        field_transformer_layers=1,
        field_transformer_heads=8,
        field_ffn_dim=1024,
    ):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.field_embeddings = nn.Embedding(num_fields, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=field_transformer_heads,
            dim_feedforward=field_ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.field_transformer = nn.TransformerEncoder(encoder_layer, num_layers=field_transformer_layers)
        self.field_attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.config = self.encoder.config
        self.config.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, field_mask=None, labels=None, **kwargs):
        if input_ids.ndim != 3 or attention_mask.ndim != 3:
            raise ValueError("FieldAwarePhoBERTClassifier expects input_ids and attention_mask shaped [B, F, L].")
        batch_size, num_fields, seq_len = input_ids.shape
        if field_mask is None:
            field_mask = torch.ones((batch_size, num_fields), dtype=torch.long, device=input_ids.device)
        else:
            field_mask = field_mask.clone()
        empty_rows = field_mask.sum(dim=1) == 0
        if empty_rows.any():
            field_mask[empty_rows, 0] = 1

        flat_input_ids = input_ids.reshape(batch_size * num_fields, seq_len)
        flat_attention_mask = attention_mask.reshape(batch_size * num_fields, seq_len)
        outputs = self.encoder(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        field_repr = outputs.last_hidden_state[:, 0].reshape(batch_size, num_fields, -1)

        field_ids = torch.arange(num_fields, device=input_ids.device).unsqueeze(0).expand(batch_size, num_fields)
        field_repr = field_repr + self.field_embeddings(field_ids)
        field_context = self.field_transformer(field_repr, src_key_padding_mask=field_mask == 0)

        field_scores = self.field_attn(field_context).squeeze(-1)
        field_scores = field_scores.masked_fill(field_mask == 0, torch.finfo(field_scores.dtype).min)
        field_weights = torch.softmax(field_scores, dim=-1)
        patient_repr = torch.sum(field_context * field_weights.unsqueeze(-1), dim=1)
        logits = self.classifier(self.dropout(patient_repr))
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits, "field_weights": field_weights}


class PhoBERTMultiTask(nn.Module):
    base_model_prefix = "encoder"

    def __init__(self, model_name="vinai/phobert-base", dropout=0.1, pooling="cls"):
        super().__init__()
        from transformers import AutoModel

        if pooling not in {"cls", "attention"}:
            raise ValueError(f"Unsupported pooling method: {pooling}")
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.pooling = pooling
        self.attn_pool = AttentionPooling(hidden_size, dropout=dropout) if pooling == "attention" else None
        self.dropout = nn.Dropout(dropout)
        self.binary_head = nn.Linear(hidden_size, 2)
        self.aux_head = nn.Linear(hidden_size, 3)
        self.config = self.encoder.config

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, aux_labels=None, lambda_aux=0.5, **kwargs):
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**model_inputs)
        hidden = outputs.last_hidden_state
        if self.pooling == "attention":
            pooled, attn_weights = self.attn_pool(hidden, attention_mask)
        else:
            pooled, attn_weights = hidden[:, 0], None
        pooled = self.dropout(pooled)
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
            loss = model(**inputs, lambda_aux=lambda_aux)["loss"].mean()
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
def eval_text(model, loader, device, threshold=0.5, desc="Evaluating", label_names=None, binary_positive_label=None, return_field_weights=False):
    model.eval()
    total_loss, total_count = 0.0, 0
    ids, labels_all, probs_all = [], [], []
    field_weights_all = []
    for batch in tqdm(loader, desc=desc, leave=False):
        inputs, batch_ids = batch_to_device(batch, device, "id")
        inputs.pop("sample_weight", None)
        labels = inputs["labels"]
        outputs = model(**inputs)
        loss = hf_loss(outputs, labels)
        probs = torch.softmax(_output_logits(outputs), dim=-1)
        if return_field_weights and isinstance(outputs, dict) and outputs.get("field_weights") is not None:
            field_weights_all.extend(outputs["field_weights"].detach().cpu().numpy().tolist())
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
    if return_field_weights:
        return (*result, np.array(field_weights_all, dtype=np.float32) if field_weights_all else None)
    return result


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
        loss = outputs["loss"].mean()
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
