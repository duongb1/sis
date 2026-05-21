import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from sislib.common import get_device, resolve_max_len, round_float, round_metrics, seed_all, split_records, to_device, unwrap
from sislib.dual_text import DualStreamPhoBERTMRIAlign
from sislib.metrics import cls_metrics, save_preds
from sislib.mri import collect_pairs
from sislib.mri_teacher import compute_mri_features, load_mri_teacher, shuffle_teacher_for_train
from sislib.text_data import TextDataset, collect_paired_text, save_records


def parse_args():
    p = argparse.ArgumentParser(description="Train dual-stream PhoBERT with auxiliary MRI feature alignment.")
    p.add_argument("--images", default="/kaggle/input/datasets/duongb/cthsis/images")
    p.add_argument("--teacher", default="/kaggle/working/mri_classifier/best_auc_model.pt")
    p.add_argument("--student", default="vinai/phobert-base")
    p.add_argument("--out", default="/kaggle/working/paired_dual_mri_align")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch-text", type=int, default=16)
    p.add_argument("--batch-mri", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--lambda-align", type=float, default=0.05)
    p.add_argument("--align-loss", choices=["cosine", "mse"], default="cosine")
    p.add_argument("--aux-dim", type=int, default=256)
    p.add_argument("--mri-dim", type=int, default=None)
    p.add_argument("--align-warmup-epochs", type=int, default=3)
    p.add_argument("--detach-aux", action="store_true")
    p.add_argument("--shuffle-teacher", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    return p.parse_args()


def current_lambda(args, epoch):
    if args.align_warmup_epochs <= 0:
        return args.lambda_align
    return args.lambda_align * min(1.0, epoch / args.align_warmup_epochs)


def train_epoch(model, loader, optimizer, scheduler, scaler, device, args, epoch):
    model.train()
    total_loss = total_ce = total_align = 0.0
    total_count = 0
    lam = current_lambda(args, epoch)
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="Training dual align", leave=False), start=1):
        batch.pop("id")
        teacher_vec = batch.pop("teacher_mri_vec").to(device, non_blocking=True)
        labels = batch.pop("labels").to(device, non_blocking=True)
        inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(
                **inputs,
                labels=labels,
                teacher_mri_vec=teacher_vec,
                lambda_align=lam,
                align_loss=args.align_loss,
                detach_aux=args.detach_aux,
            )
            loss = outputs["loss"] / args.accum
        scaler.scale(loss).backward()
        if step % args.accum == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        bs = labels.size(0)
        total_loss += loss.item() * args.accum * bs
        total_ce += outputs["loss_ce"].detach().item() * bs
        total_align += outputs["loss_align"].detach().item() * bs
        total_count += bs
    denom = max(total_count, 1)
    return float(total_loss / denom), float(total_ce / denom), float(total_align / denom), lam


@torch.no_grad()
def eval_dual(model, loader, device, threshold):
    model.eval()
    total_loss, total_count = 0.0, 0
    ids, labels_all, probs_all = [], [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch_ids = batch.pop("id")
        batch.pop("teacher_mri_vec", None)
        labels = batch.pop("labels").to(device, non_blocking=True)
        inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        outputs = model(**inputs, labels=labels, teacher_mri_vec=None)
        probs = torch.softmax(outputs["logits"], dim=-1)[:, 1]
        bs = labels.size(0)
        total_loss += outputs["loss"].item() * bs
        total_count += bs
        ids.extend(batch_ids)
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        probs_all.extend(probs.detach().cpu().numpy().tolist())
    labels = np.array(labels_all, dtype=np.int64)
    probs = np.array(probs_all, dtype=np.float32)
    preds = (probs >= threshold).astype(np.int64)
    loss = total_loss / max(total_count, 1)
    return cls_metrics(labels, probs, preds, loss=loss, threshold=threshold), ids, labels, probs, preds


def save_best(model, tokenizer, path, epoch, metrics, args, max_len, mri_dim):
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": unwrap(model).state_dict()}, path / "model.pt")
    tokenizer.save_pretrained(path)
    with open(path / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "epoch": epoch,
                "best_model_metric": "auc",
                "val_metrics": round_metrics(metrics),
                "student_init": args.student,
                "mri_teacher": args.teacher,
                "lambda_align": args.lambda_align,
                "align_loss": args.align_loss,
                "aux_dim": args.aux_dim,
                "mri_dim": mri_dim,
                "align_warmup_epochs": args.align_warmup_epochs,
                "detach_aux": args.detach_aux,
                "shuffle_teacher": args.shuffle_teacher,
                "max_length": max_len,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    args = parse_args()
    seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.cpu)
    multi_gpu = not args.no_mgpu

    text_records, missing, image_root = collect_paired_text(args.images)
    if missing:
        (out / "missing_text_patients.txt").write_text("\n".join(missing), encoding="utf-8")
    mri_records, _ = collect_pairs(args.images)
    train_rows, val_rows, test_rows = [split_records(text_records, s) for s in ("train", "val", "test")]
    print(f"Image root: {image_root.resolve()}")
    print(f"Text patients: {len(text_records)} | Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)}")

    teacher = load_mri_teacher(args.teacher, device, multi_gpu)
    teacher_vecs = compute_mri_features(text_records, mri_records, teacher, device, args.batch_mri, args.workers)
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    train_rows = [row for row in train_rows if row["id"] in teacher_vecs]
    if args.shuffle_teacher:
        teacher_vecs = shuffle_teacher_for_train(teacher_vecs, train_rows, args.seed)
    if not train_rows or not val_rows:
        raise RuntimeError("Need non-empty train and val records.")

    mri_dim = args.mri_dim or len(next(iter(teacher_vecs.values())))
    tokenizer = AutoTokenizer.from_pretrained(args.student, use_fast=False)
    model = DualStreamPhoBERTMRIAlign(args.student, aux_dim=args.aux_dim, mri_dim=mri_dim)
    max_len = resolve_max_len(model.encoder, args.max_len)
    if max_len != args.max_len:
        print(f"Requested max_len={args.max_len}, using {max_len}.")
    model = to_device(model, device, multi_gpu)

    train_loader = DataLoader(TextDataset(train_rows, tokenizer, max_len, teacher_mri_vecs=teacher_vecs), batch_size=args.batch_text, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(TextDataset(val_rows, tokenizer, max_len), batch_size=args.batch_text, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(TextDataset(test_rows, tokenizer, max_len), batch_size=args.batch_text, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda") if test_rows else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, int(np.ceil(len(train_loader) / max(args.accum, 1))) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * args.warmup), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best, best_dir, history = -1.0, out / "best_auc_dual_mri_align", []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_ce, train_align, lam = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, args, epoch)
        val_metrics, val_ids, val_y, val_p, val_pred = eval_dual(model, val_loader, device, args.threshold)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round_float(train_loss),
                "train_loss_ce": round_float(train_ce),
                "train_loss_align": round_float(train_align),
                "lambda_align": round_float(lam),
                "val_loss": round_float(val_metrics["loss"]),
                "val_acc": round_float(val_metrics["accuracy"]),
                "val_f1": round_float(val_metrics["f1"]),
                "val_auc": round_float(val_metrics["auc"]),
            }
        )
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.3f} | ce={train_ce:.3f} | align={train_align:.3f} | lambda={lam:.3f} | val_auc={val_metrics['auc']:.3f}")
        if val_metrics["auc"] > best:
            best = val_metrics["auc"]
            save_best(model, tokenizer, best_dir, epoch, val_metrics, args, max_len, mri_dim)
            save_preds(out / "val_predictions_best_auc.csv", val_ids, val_y, val_p, val_pred, "id")

    with open(out / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    save_records(out / "dataset_records.csv", text_records)

    ckpt = torch.load(best_dir / "model.pt", map_location=device, weights_only=False)
    unwrap(model).load_state_dict(ckpt["model_state"])
    all_metrics = {}
    for split, loader in [("val", val_loader), ("test", test_loader)]:
        if loader is None:
            continue
        metrics, ids, y, p, pred = eval_dual(model, loader, device, args.threshold)
        all_metrics[split] = round_metrics(metrics)
        save_preds(out / f"{split}_predictions_best_auc.csv", ids, y, p, pred, "id")
        print(f"{split}: {round_metrics(metrics)}")
        print(confusion_matrix(y, pred, labels=[0, 1]))
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
