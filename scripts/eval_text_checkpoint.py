import argparse
import json
from pathlib import Path

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sislib.common import get_device, resolve_max_len, round_metrics, seed_all, to_device
from sislib.metrics import save_preds
from sislib.text_data import TextDataset, collect_large_text, collect_paired_text, save_records
from sislib.text_train import eval_text


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a saved text checkpoint on a selected SIS text test split.")
    p.add_argument("--checkpoint", required=True, help="Path to a saved Hugging Face text checkpoint directory.")
    p.add_argument("--dataset", choices=["large", "paired"], required=True)
    p.add_argument("--texts", default="/kaggle/input/datasets/duongb/cthsis/texts")
    p.add_argument("--images", default="/kaggle/input/datasets/duongb/cthsis/images")
    p.add_argument("--out", required=True)
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    return p.parse_args()


def load_records(args):
    if args.dataset == "large":
        records, skipped, data_root = collect_large_text(args.texts, splits=("test",))
        return records, skipped, data_root
    records, missing, image_root = collect_paired_text(args.images, splits=("test",))
    return records, missing, image_root


def checkpoint_max_len(checkpoint):
    info_path = Path(checkpoint) / "training_info.json"
    if not info_path.exists():
        return None
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    return info.get("max_length")


def main():
    args = parse_args()
    seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records, skipped, root = load_records(args)
    if skipped:
        (out / "skipped_or_missing_text_files.txt").write_text("\n".join(skipped), encoding="utf-8")
    if not records:
        raise RuntimeError("Need non-empty test records.")

    device = get_device(args.cpu)
    checkpoint = Path(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    requested_max_len = args.max_len or checkpoint_max_len(checkpoint) or 512
    max_len = resolve_max_len(model, requested_max_len)
    if max_len != requested_max_len:
        print(f"Requested max_len={requested_max_len}, using {max_len}.")
    model = to_device(model, device, not args.no_mgpu)

    print(f"Checkpoint: {checkpoint.resolve()}")
    print(f"Dataset root: {root.resolve()}")
    print(f"Dataset: {args.dataset} | Test samples: {len(records)}")

    test_loader = DataLoader(
        TextDataset(records, tokenizer, max_len),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    metrics, ids, y, p, pred = eval_text(model, test_loader, device, args.threshold, desc=f"Evaluating {args.dataset} test")
    rounded = round_metrics(metrics)
    save_preds(out / "test_predictions.csv", ids, y, p, pred, "id")
    save_records(out / "dataset_records.csv", records)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"test": rounded}, f, ensure_ascii=False, indent=2)
    print(f"test: {rounded}")
    print(confusion_matrix(y, pred, labels=[0, 1]))


if __name__ == "__main__":
    main()
