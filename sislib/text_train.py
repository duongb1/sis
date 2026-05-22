import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .common import batch_to_device, unwrap
from .metrics import cls_metrics


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


@torch.no_grad()
def eval_text(model, loader, device, threshold=0.5, desc="Evaluating"):
    model.eval()
    total_loss, total_count = 0.0, 0
    ids, labels_all, probs_all = [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        inputs, batch_ids = batch_to_device(batch, device, "id")
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
