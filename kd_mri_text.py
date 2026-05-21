import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from sislib.common import get_device, resolve_max_len, round_float, round_metrics, seed_all, split_records, to_device, unwrap
from sislib.metrics import save_preds
from sislib.mri import MRIDataset, collect_pairs, resnet50_binary
from sislib.text_data import TextDataset, collect_paired_text, save_records
from sislib.text_train import eval_text, kd_epoch, kd_loss_fn


def parse_args():
    p = argparse.ArgumentParser(description="MRI teacher -> paired-text student KD.")
    p.add_argument("--images", default="/kaggle/input/datasets/duongb/cthsis/images")
    p.add_argument("--teacher", default="/kaggle/working/mri_classifier/best_auc_model.pt")
    p.add_argument("--student", default="/kaggle/working/text_phobert_classifier/best_auc_phobert")
    p.add_argument("--out", default="/kaggle/working/paired_text_kd_from_mri_teacher")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch-text", type=int, default=16)
    p.add_argument("--batch-mri", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--kd", choices=["binary", "kl"], default="binary")
    p.add_argument("--temp", type=float, default=2.0)
    p.add_argument("--ce-warmup", type=int, default=0)
    p.add_argument("--teacher-correct-only", action="store_true")
    p.add_argument("--shuffle-teacher", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    return p.parse_args()


def clean_state_dict(state):
    return {k.replace("module.", "", 1): v for k, v in state.items()} if any(k.startswith("module.") for k in state) else state


def load_mri(path, device, multi_gpu):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model = resnet50_binary()
    model.load_state_dict(clean_state_dict(state))
    return to_device(model.eval(), device, multi_gpu)


@torch.no_grad()
def mri_teacher_logits(text_records, mri_records, teacher, device, batch_size, workers):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(MRIDataset(mri_records, tf), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    by_id = {}
    for images, _, ids in tqdm(loader, desc="Computing MRI teacher logits"):
        logits = teacher(images.to(device, non_blocking=True)).squeeze(1).detach().cpu().numpy()
        for logit, item_id in zip(logits, ids):
            by_id.setdefault(item_id, []).append(float(logit))

    text_ids = {r["id"] for r in text_records}
    logits_2 = {}
    for item_id, vals in by_id.items():
        if item_id not in text_ids:
            continue
        z = float(np.mean(vals))
        logits_2[item_id] = [-z / 2.0, z / 2.0]
    return logits_2


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
    return {
        "teacher_accuracy": float(accuracy_score(labels, preds)) if len(labels) else float("nan"),
        "teacher_f1": float(f1_score(labels, preds, zero_division=0)) if len(labels) else float("nan"),
        "teacher_auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else float("nan"),
        "teacher_num_patients": int(len(labels)),
        "teacher_correct_patients": int((preds == labels).sum()) if len(labels) else 0,
        "teacher_missing_patients": int(len(records) - len(rows)),
    }


def teacher_correct_ids(records, teacher_logits, threshold):
    correct = set()
    for row in records:
        if row["id"] not in teacher_logits:
            continue
        logits = np.array(teacher_logits[row["id"]], dtype=np.float32)
        exp_logits = np.exp(logits - logits.max())
        p_co = exp_logits[1] / exp_logits.sum()
        if int(p_co >= threshold) == row["label"]:
            correct.add(row["id"])
    return correct


def shuffle_teacher_for_train(teacher_logits, train_rows, seed):
    rng = random.Random(seed)
    ids = [row["id"] for row in train_rows if row["id"] in teacher_logits]
    shuffled = ids[:]
    rng.shuffle(shuffled)
    out = dict(teacher_logits)
    for target_id, source_id in zip(ids, shuffled):
        out[target_id] = teacher_logits[source_id]
    return out


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

    teacher = load_mri(args.teacher, device, multi_gpu)
    logits = mri_teacher_logits(text_records, mri_records, teacher, device, args.batch_mri, args.workers)
    stats = {
        "train": round_metrics(teacher_stats_for_records(train_rows, logits, args.threshold)),
        "val": round_metrics(teacher_stats_for_records(val_rows, logits, args.threshold)),
        "test": round_metrics(teacher_stats_for_records(test_rows, logits, args.threshold)),
    }
    if args.teacher_correct_only:
        correct = teacher_correct_ids(train_rows, logits, args.threshold)
        train_rows = [r for r in train_rows if r["id"] in logits]
        train_rows = [r for r in train_rows if r["id"] in correct]
    else:
        train_rows = [r for r in train_rows if r["id"] in logits]
    if args.shuffle_teacher:
        logits = shuffle_teacher_for_train(logits, train_rows, args.seed)
    if not train_rows or not val_rows:
        raise RuntimeError("Need non-empty train and val records.")
    print(f"Teacher train stats: {stats['train']}")
    print(f"Teacher val stats: {stats['val']}")
    print(f"Teacher test stats: {stats['test']}")
    print(f"KD train rows with MRI signal: {len(train_rows)}")
    with open(out / "teacher_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained(args.student, use_fast=False)
    student = AutoModelForSequenceClassification.from_pretrained(args.student, num_labels=2, id2label={0: "khong", 1: "co"}, label2id={"khong": 0, "co": 1})
    max_len = resolve_max_len(student, args.max_len)
    if max_len != args.max_len:
        print(f"Requested max_len={args.max_len}, using {max_len}.")
    student = to_device(student, device, multi_gpu)

    train_loader = DataLoader(TextDataset(train_rows, tok, max_len, logits), batch_size=args.batch_text, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(TextDataset(val_rows, tok, max_len), batch_size=args.batch_text, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(TextDataset(test_rows, tok, max_len), batch_size=args.batch_text, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda") if test_rows else None

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, int(np.ceil(len(train_loader) / max(args.accum, 1))) * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(steps * args.warmup), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = kd_loss_fn(args.alpha, args.kd, args.temp, args.ce_warmup)
    best, best_dir, history = -1.0, out / "best_auc_mri_text_kd", []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ce, tr_kd = kd_epoch(student, train_loader, opt, sched, scaler, device, loss_fn, epoch, args.accum)
        val_metrics, val_ids, val_y, val_p, val_pred = eval_text(student, val_loader, device, args.threshold)
        history.append({"epoch": epoch, "train_loss": round_float(tr_loss), "train_ce": round_float(tr_ce), "train_kd": round_float(tr_kd), "val_loss": round_float(val_metrics["loss"]), "val_acc": round_float(val_metrics["accuracy"]), "val_f1": round_float(val_metrics["f1"]), "val_auc": round_float(val_metrics["auc"])})
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={tr_loss:.3f} | val_auc={val_metrics['auc']:.3f}")
        if val_metrics["auc"] > best:
            best = val_metrics["auc"]
            best_dir.mkdir(parents=True, exist_ok=True)
            unwrap(student).save_pretrained(best_dir)
            tok.save_pretrained(best_dir)
            with open(best_dir / "training_info.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "best_model_metric": "auc",
                        "val_metrics": round_metrics(val_metrics),
                        "student_init": args.student,
                        "mri_teacher": args.teacher,
                        "alpha": args.alpha,
                        "kd_loss": args.kd,
                        "temperature": args.temp,
                        "teacher_correct_only": args.teacher_correct_only,
                        "shuffle_teacher": args.shuffle_teacher,
                        "max_length": max_len,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            save_preds(out / "val_predictions_best_auc.csv", val_ids, val_y, val_p, val_pred, "id")
    with open(out / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    save_records(out / "dataset_records.csv", text_records)

    eval_model = to_device(AutoModelForSequenceClassification.from_pretrained(best_dir), device, multi_gpu)
    all_metrics = {}
    for split, loader in [("val", val_loader), ("test", test_loader)]:
        if loader is None:
            continue
        metrics, ids, y, p, pred = eval_text(eval_model, loader, device, args.threshold)
        all_metrics[split] = round_metrics(metrics)
        save_preds(out / f"{split}_predictions_best_auc.csv", ids, y, p, pred, "id")
        print(f"{split}: {round_metrics(metrics)}")
        print(confusion_matrix(y, pred, labels=[0, 1]))
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
