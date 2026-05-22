import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .common import batch_to_device, unwrap
from .metrics import cls_metrics


def hf_loss(outputs, labels):
    loss = outputs.loss if outputs.loss is not None else F.cross_entropy(outputs.logits, labels)
    return loss.mean()


def ce_epoch(model, loader, optimizer, scheduler, scaler, device, accum=1, class_weights=None):
    model.train()
    total_loss, total_count = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="Training", leave=False), start=1):
        inputs, _ = batch_to_device(batch, device, "id")
        labels = inputs["labels"]
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            if class_weights is None:
                loss = hf_loss(model(**inputs), labels)
            else:
                model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
                loss = F.cross_entropy(model(**model_inputs).logits, labels, weight=class_weights)
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


def kd_epoch(model, loader, optimizer, scheduler, scaler, device, loss_fn, epoch, accum=1):
    model.train()
    total_loss = total_ce = total_kd = 0.0
    total_count = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="Training KD", leave=False), start=1):
        inputs, _ = batch_to_device(batch, device, "id")
        labels = inputs["labels"]
        teacher_logits = inputs.pop("teacher_logits")
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss, ce, kd = loss_fn(model(**inputs).logits, teacher_logits, labels, epoch)
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
        total_ce += ce.item() * bs
        total_kd += kd.item() * bs
        total_count += bs
    denom = max(total_count, 1)
    return float(total_loss / denom), float(total_ce / denom), float(total_kd / denom)


@torch.no_grad()
def eval_text(model, loader, device, threshold=0.5, desc="Evaluating"):
    model.eval()
    total_loss, total_count = 0.0, 0
    ids, labels_all, probs_all = [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        inputs, batch_ids = batch_to_device(batch, device, "id")
        inputs.pop("teacher_logits", None)
        inputs.pop("teacher_mri_vec", None)
        inputs.pop("sample_weight", None)
        labels = inputs["labels"]
        outputs = model(**inputs)
        loss = hf_loss(outputs, labels)
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_count += bs
        ids.extend(batch_ids)
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        probs_all.extend(probs.detach().cpu().numpy().tolist())
    labels = np.array(labels_all, dtype=np.int64)
    probs = np.array(probs_all, dtype=np.float32)
    preds = (probs >= threshold).astype(np.int64)
    loss = total_loss / max(total_count, 1)
    return cls_metrics(labels, probs, preds, loss=loss, threshold=threshold), ids, labels, probs, preds


def kd_loss_fn(alpha, kind="binary", temperature=2.0, warmup=0, kd_weight="none"):
    def loss_fn(student_logits, teacher_logits, labels, epoch):
        ce = F.cross_entropy(student_logits, labels, reduction="none")
        if epoch <= warmup or alpha <= 0:
            zero = torch.zeros((), device=student_logits.device)
            return ce.mean(), ce.mean().detach(), zero
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        if kind == "binary":
            teacher_p = teacher_probs[:, 1].detach()
            student_logit = student_logits[:, 1] - student_logits[:, 0]
            kd = F.binary_cross_entropy_with_logits(student_logit, teacher_p, reduction="none")
        elif kind == "kl":
            kd = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="none",
            ).sum(dim=-1) * (temperature ** 2)
        else:
            raise ValueError(f"Unsupported KD loss: {kind}")
        if kd_weight == "none":
            weights = torch.ones_like(kd)
        elif kd_weight == "confidence":
            teacher_conf = teacher_probs.max(dim=1).values.detach()
            weights = torch.clamp((teacher_conf - 0.5) * 2.0, min=0.0, max=1.0)
        else:
            raise ValueError(f"Unsupported KD weight: {kd_weight}")
        weighted_kd = weights * kd
        loss = (1.0 - alpha) * ce + alpha * weighted_kd
        return loss.mean(), ce.mean().detach(), weighted_kd.mean().detach()
    return loss_fn
