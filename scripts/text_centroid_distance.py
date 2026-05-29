import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sislib.common import SPLITS, get_device, quiet_hf_logging, resolve_max_len, seed_all

quiet_hf_logging()

from transformers import AutoModel, AutoTokenizer

from sislib.text_data import TextDataset, collect_large_text, collect_paired_text


def parse_args():
    p = argparse.ArgumentParser(description="Compute class-wise text embedding centroid distances for large vs paired text.")
    p.add_argument("--texts", default="/kaggle/input/datasets/duongb/cthsis/data/texts")
    p.add_argument("--images", default="/kaggle/input/datasets/duongb/cthsis/data/images")
    p.add_argument("--out", default="/kaggle/working/sis_runs/05_text_centroids")
    p.add_argument("--checkpoint", default=None, help="Optional Hugging Face checkpoint directory.")
    p.add_argument("--model", default="vinai/phobert-base", help="Base model used when --checkpoint is missing.")
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    p.add_argument("--sample-per-group", type=int, default=0, help="Max samples per dataset/split/label; 0 keeps all.")
    p.add_argument("--no-normalize", action="store_true", help="Do not L2-normalize embeddings before centroids.")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def checkpoint_max_len(checkpoint):
    if not checkpoint:
        return None
    info_path = Path(checkpoint) / "training_info.json"
    if not info_path.exists():
        return None
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f).get("max_length")


def resolve_model_source(args):
    if not args.checkpoint:
        return args.model, "base_model"

    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.exists():
        print(
            f"Checkpoint not found: {checkpoint}\n"
            f"Falling back to base model for embedding extraction: {args.model}",
            file=sys.stderr,
        )
        return args.model, "fallback_base_model"
    if checkpoint.is_file():
        raise NotADirectoryError(f"--checkpoint must be a Hugging Face checkpoint directory, got: {checkpoint}")
    if not (checkpoint / "config.json").exists():
        print(
            f"Checkpoint directory is missing config.json: {checkpoint}\n"
            f"Falling back to base model for embedding extraction: {args.model}",
            file=sys.stderr,
        )
        return args.model, "fallback_base_model"
    return str(checkpoint), "checkpoint"


def balanced_sample(records, limit, seed):
    if limit <= 0:
        return records
    rng = np.random.default_rng(seed)
    sampled = []
    for dataset in ("large", "paired"):
        for split in SPLITS:
            for label in (0, 1):
                group = [r for r in records if r["dataset"] == dataset and r["split"] == split and r["label"] == label]
                if len(group) > limit:
                    idx = rng.choice(len(group), size=limit, replace=False)
                    group = [group[i] for i in sorted(idx)]
                sampled.extend(group)
    return sampled


def load_records(args):
    splits = tuple(args.splits)
    large_records, large_skipped, large_root = collect_large_text(args.texts, splits=splits)
    paired_records, paired_missing, paired_root = collect_paired_text(args.images, splits=splits)
    for row in large_records:
        row["dataset"] = "large"
        row["plot_id"] = f"large/{row['id']}"
    for row in paired_records:
        row["dataset"] = "paired"
        row["plot_id"] = f"paired/{row['id']}"
    records = balanced_sample(large_records + paired_records, args.sample_per_group, args.seed)
    return records, large_skipped, paired_missing, large_root, paired_root


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    ids, vectors = [], []
    for batch in loader:
        batch_ids = batch.pop("id")
        batch.pop("labels", None)
        inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        outputs = model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :]
        ids.extend(batch_ids)
        vectors.append(cls.detach().cpu().numpy())
    return ids, np.concatenate(vectors, axis=0)


def group_key(row):
    suffix = "pos" if row["label"] == 1 else "neg"
    return f"{row['dataset']}_{suffix}"


def compute_centroids(records, embeddings):
    centroids = {}
    counts = {}
    for key in ("large_pos", "large_neg", "paired_pos", "paired_neg"):
        idx = [i for i, row in enumerate(records) if group_key(row) == key]
        if not idx:
            raise RuntimeError(f"Cannot compute centroid for empty group: {key}")
        centroids[key] = embeddings[idx].mean(axis=0, keepdims=True)
        counts[key] = len(idx)
    return centroids, counts


def compute_distances(centroids):
    pairs = [
        ("large_pos vs paired_pos", "large_pos", "paired_pos"),
        ("large_neg vs paired_neg", "large_neg", "paired_neg"),
        ("large_pos vs large_neg", "large_pos", "large_neg"),
        ("paired_pos vs paired_neg", "paired_pos", "paired_neg"),
        ("large_pos vs paired_neg", "large_pos", "paired_neg"),
        ("large_neg vs paired_pos", "large_neg", "paired_pos"),
    ]
    rows = []
    for name, left, right in pairs:
        rows.append(
            {
                "pair": name,
                "left": left,
                "right": right,
                "cosine_distance": float(cosine_distances(centroids[left], centroids[right])[0, 0]),
                "euclidean_distance": float(euclidean_distances(centroids[left], centroids[right])[0, 0]),
            }
        )
    return rows


def save_distances(path, rows):
    fieldnames = ["pair", "left", "right", "cosine_distance", "euclidean_distance"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_group_counts(path, counts):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "count"])
        writer.writeheader()
        for key, count in counts.items():
            writer.writerow({"group": key, "count": count})


def main():
    args = parse_args()
    seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records, large_skipped, paired_missing, large_root, paired_root = load_records(args)
    if large_skipped:
        (out / "large_skipped_text_files.txt").write_text("\n".join(large_skipped), encoding="utf-8")
    if paired_missing:
        (out / "paired_missing_text_files.txt").write_text("\n".join(paired_missing), encoding="utf-8")
    if not records:
        raise RuntimeError("Need non-empty large/paired text records.")

    model_name_or_path, model_source = resolve_model_source(args)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = AutoModel.from_pretrained(model_name_or_path)
    requested_max_len = args.max_len or checkpoint_max_len(args.checkpoint) or 512
    max_len = resolve_max_len(model, requested_max_len)

    device = get_device(args.cpu)
    model = model.to(device)
    loader = DataLoader(
        TextDataset(records, tokenizer, max_len),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    ids, embeddings = extract_embeddings(model, loader, device)
    if ids != [row["id"] for row in records]:
        raise RuntimeError("Embedding order mismatch.")

    if not args.no_normalize:
        embeddings = normalize(embeddings)

    centroids, counts = compute_centroids(records, embeddings)
    distances = compute_distances(centroids)

    np.save(out / "text_embeddings_for_centroids.npy", embeddings)
    np.savez(out / "text_centroids.npz", **{key: value.squeeze(0) for key, value in centroids.items()})
    save_distances(out / "centroid_distances.csv", distances)
    save_group_counts(out / "centroid_group_counts.csv", counts)

    summary = {
        "model": str(Path(model_name_or_path).resolve()) if Path(model_name_or_path).exists() else model_name_or_path,
        "model_source": model_source,
        "requested_checkpoint": args.checkpoint,
        "large_root": str(Path(large_root).resolve()),
        "paired_root": str(Path(paired_root).resolve()),
        "splits": args.splits,
        "max_len": max_len,
        "normalized_embeddings": not args.no_normalize,
        "samples": len(records),
        "group_counts": counts,
        "distances": distances,
    }
    with open(out / "centroid_distance_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Class-wise centroid distances")
    print(f"Model: {summary['model']} ({model_source})")
    print(f"Splits: {', '.join(args.splits)}")
    print(f"Normalized embeddings: {summary['normalized_embeddings']}")
    for row in distances:
        print(
            f"{row['pair']}: "
            f"cosine={row['cosine_distance']:.4f}, "
            f"euclidean={row['euclidean_distance']:.4f}"
        )
    print(f"Saved: {(out / 'centroid_distances.csv').resolve()}")
    print(f"Saved: {(out / 'centroid_distance_summary.json').resolve()}")


if __name__ == "__main__":
    main()
