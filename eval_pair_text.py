import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sislib.common import get_device, resolve_max_len, round_metrics, seed_all, split_records, to_device
from sislib.metrics import save_preds
from sislib.text_data import TextDataset, collect_paired_text, save_records
from sislib.text_train import eval_text


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a text checkpoint directly on paired text splits.")
    p.add_argument("--images", default="/kaggle/input/datasets/duongb/cthsis/images")
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="/kaggle/working/pair_text_eval")
    p.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records, missing, image_root = collect_paired_text(args.images)
    if missing:
        (out / "missing_text_files.txt").write_text("\n".join(missing), encoding="utf-8")

    device = get_device(args.cpu)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    max_len = resolve_max_len(model, args.max_len)
    if max_len != args.max_len:
        print(f"Requested max_len={args.max_len}, using {max_len}.")
    model = to_device(model, device, not args.no_mgpu)

    print(f"Image root: {image_root.resolve()}")
    print(f"Paired text samples: {len(records)}")

    all_metrics = {}
    for split in args.splits:
        rows = split_records(records, split)
        loader = DataLoader(
            TextDataset(rows, tokenizer, max_len),
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        metrics, ids, y, p, pred = eval_text(model, loader, device, args.threshold, desc=f"Evaluating {split}")
        all_metrics[split] = round_metrics(metrics)
        save_preds(out / f"{split}_predictions_best_auc.csv", ids, y, p, pred, "id")
        print(f"{split}: {round_metrics(metrics)}")
        print(confusion_matrix(y, pred, labels=[0, 1]))

    save_records(out / "dataset_records.csv", records)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
