import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from sislib.common import quiet_hf_logging

quiet_hf_logging()

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sislib.common import get_device, resolve_max_len, round_metrics, seed_all, to_device
from sislib.metrics import save_preds
from sislib.text_data import TextDataset, collect_large_text, collect_paired_text, collect_small_text, make_label_maps, save_records
from sislib.text_train import eval_text


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a saved text checkpoint on a selected SIS text test split.")
    p.add_argument("--checkpoint", required=True, help="Path to a saved Hugging Face text checkpoint directory.")
    p.add_argument("--dataset", choices=["large", "small", "paired"], required=True)
    p.add_argument("--texts", default="/kaggle/input/datasets/duongb/cthsis/data/texts")
    p.add_argument("--small", default="/kaggle/input/datasets/duongb/cthsis/sis/small")
    p.add_argument("--images", default="/kaggle/input/datasets/duongb/cthsis/data/images")
    p.add_argument("--out", required=True)
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--binary-positive-label", default=None, help="Class treated as positive for one-vs-rest binary metrics. Defaults to checkpoint value or I63_INFARCTION.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-mgpu", action="store_true")
    return p.parse_args()


def load_checkpoint_info(checkpoint):
    info_path = Path(checkpoint) / "training_info.json"
    if not info_path.exists():
        return {}
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_labels(checkpoint, model=None):
    info = load_checkpoint_info(checkpoint)
    labels = info.get("labels")
    label_to_id = info.get("label_to_id")
    if labels and label_to_id:
        return list(labels), {str(k): int(v) for k, v in label_to_id.items()}
    if model is not None and getattr(model.config, "id2label", None):
        pairs = sorted((int(index), label) for index, label in model.config.id2label.items())
        labels = [label for _, label in pairs]
        label_to_id, _ = make_label_maps(labels)
        return labels, label_to_id
    return None, None


def load_records(args, labels=None, label_to_id=None):
    if args.dataset == "large":
        records, skipped, data_root = collect_large_text(args.texts, splits=("test",), labels=labels, label_to_id=label_to_id)
        return records, skipped, data_root
    if args.dataset == "small":
        records, skipped, data_root = collect_small_text(args.small, splits=("test",), labels=labels, label_to_id=label_to_id)
        return records, skipped, data_root
    records, missing, image_root = collect_paired_text(args.images, splits=("test",))
    return records, missing, image_root


def checkpoint_max_len(checkpoint):
    return load_checkpoint_info(checkpoint).get("max_length")


def checkpoint_binary_positive_label(checkpoint, requested):
    if requested:
        return requested
    return load_checkpoint_info(checkpoint).get("binary_positive_label") or "I63_INFARCTION"


def main():
    args = parse_args()
    seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = get_device(args.cpu)
    checkpoint = Path(args.checkpoint)
    binary_positive_label = checkpoint_binary_positive_label(checkpoint, args.binary_positive_label)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    labels, label_to_id = load_checkpoint_labels(checkpoint, model)
    records, skipped, root = load_records(args, labels=labels, label_to_id=label_to_id)
    if skipped:
        (out / "skipped_or_missing_text_files.txt").write_text("\n".join(skipped), encoding="utf-8")
    if not records:
        raise RuntimeError("Need non-empty test records.")

    requested_max_len = args.max_len or checkpoint_max_len(checkpoint) or 512
    max_len = resolve_max_len(model, requested_max_len)
    if max_len != requested_max_len:
        print(f"Requested max_len={requested_max_len}, using {max_len}.")
    model = to_device(model, device, not args.no_mgpu)

    print(f"Checkpoint: {checkpoint.resolve()}")
    print(f"Dataset root: {root.resolve()}")
    print(f"Dataset: {args.dataset} | Test samples: {len(records)}")
    if labels:
        print(f"Labels ({len(labels)}): {', '.join(labels)}")

    test_loader = DataLoader(
        TextDataset(records, tokenizer, max_len),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    metrics, ids, y, p, pred = eval_text(
        model,
        test_loader,
        device,
        args.threshold,
        desc=f"Evaluating {args.dataset} test",
        label_names=labels,
        binary_positive_label=binary_positive_label,
    )
    rounded = round_metrics(metrics)
    save_preds(
        out / "test_predictions.csv",
        ids,
        y,
        p,
        pred,
        "id",
        label_names=labels,
        binary_positive_label=binary_positive_label,
    )
    save_records(out / "dataset_records.csv", records)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"test": rounded}, f, ensure_ascii=False, indent=2)
    print(f"test: {rounded}")
    print(confusion_matrix(y, pred, labels=list(range(len(labels))) if labels else None))


if __name__ == "__main__":
    main()
