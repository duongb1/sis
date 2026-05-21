import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sislib.common import get_device, resolve_max_len, round_metrics, to_device
from sislib.metrics import save_preds
from sislib.text_data import TextDataset, collect_paired_text, save_records
from sislib.text_train import eval_text


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate text checkpoint on image-test patient txt files.")
    p.add_argument("--images", default="/kaggle/input/datasets/duongb/cthsis/images")
    p.add_argument("--model", default="/kaggle/working/text_phobert_classifier/best_auc_phobert")
    p.add_argument("--out", default="/kaggle/working/text_phobert_classifier/image_test_patients")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.cpu)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    max_len = resolve_max_len(model, args.max_len)
    model = to_device(model, device, not args.no_mgpu)

    records, missing, image_root = collect_paired_text(args.images, splits=["test"])
    if missing:
        (out / "missing_text_patients.txt").write_text("\n".join(missing), encoding="utf-8")
    if not records:
        raise RuntimeError("No test text records found.")
    print(f"Image root: {image_root.resolve()}")
    print(f"Image-test text patients: {len(records)} | Missing/empty: {len(missing)}")

    loader = DataLoader(TextDataset(records, tokenizer, max_len), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    metrics, ids, y, p, pred = eval_text(model, loader, device, args.threshold, desc="Evaluating image-test text")
    metrics = round_metrics(metrics)
    save_preds(out / "image_test_text_predictions.csv", ids, y, p, pred, "id")
    save_records(out / "image_test_text_records.csv", records)
    with open(out / "image_test_text_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics: {metrics}")
    print(confusion_matrix(y, pred, labels=[0, 1]))


if __name__ == "__main__":
    main()
