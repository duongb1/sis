import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sislib.common import SPLITS, get_device, quiet_hf_logging, resolve_max_len, seed_all

quiet_hf_logging()

from transformers import AutoModel, AutoTokenizer

from sislib.text_data import TextDataset, collect_large_text, collect_paired_text


def parse_args():
    p = argparse.ArgumentParser(description="Plot t-SNE for large text and paired text across train/val/test splits.")
    p.add_argument("--texts", default="/kaggle/input/datasets/duongb/cthsis/texts")
    p.add_argument("--images", default="/kaggle/input/datasets/duongb/cthsis/images")
    p.add_argument("--out", default="/kaggle/working/sis_runs/04_text_tsne")
    p.add_argument("--checkpoint", default=None, help="Optional Hugging Face checkpoint directory to extract embeddings from.")
    p.add_argument("--model", default="vinai/phobert-base", help="Base model used when --checkpoint is not set.")
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--sample-per-group", type=int, default=0, help="Max samples per dataset/split/label; 0 keeps all.")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def checkpoint_max_len(checkpoint):
    info_path = Path(checkpoint) / "training_info.json"
    if not info_path.exists():
        return None
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f).get("max_length")


def resolve_model_source(args):
    if not args.checkpoint:
        return args.model

    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint}\n"
            "Transformers will treat a missing absolute path as a Hugging Face repo id, which causes a confusing "
            "HFValidationError. Check that the training stage created this directory, or pass the correct "
            "--checkpoint path. For the default pipeline, expected paths are usually:\n"
            "  /kaggle/working/sis_runs/01_large_text_ce/best_auc_phobert\n"
            "  /kaggle/working/sis_runs/02_paired_text_ce/best_auc_phobert"
        )
    if not checkpoint.is_dir():
        raise NotADirectoryError(f"--checkpoint must be a Hugging Face checkpoint directory, got: {checkpoint}")
    if not (checkpoint / "config.json").exists():
        raise FileNotFoundError(f"Checkpoint directory is missing config.json: {checkpoint}")
    return str(checkpoint)


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
    large_records, large_skipped, large_root = collect_large_text(args.texts, splits=SPLITS)
    paired_records, paired_missing, paired_root = collect_paired_text(args.images, splits=SPLITS)
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
    for batch in tqdm(loader, desc="Extracting text embeddings", leave=False):
        batch_ids = batch.pop("id")
        batch.pop("labels", None)
        inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        outputs = model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :]
        ids.extend(batch_ids)
        vectors.append(cls.detach().cpu().numpy())
    return ids, np.concatenate(vectors, axis=0)


def fit_tsne(embeddings, seed, requested_perplexity):
    if len(embeddings) < 2:
        raise RuntimeError("Need at least 2 samples for t-SNE.")
    perplexity = min(requested_perplexity, max(1.0, (len(embeddings) - 1) / 3.0))
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=seed)
    return tsne.fit_transform(embeddings), perplexity


def save_points(path, records, points):
    fieldnames = ["plot_id", "dataset", "split", "label", "label_name", "text_path", "tsne_x", "tsne_y"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, xy in zip(records, points):
            writer.writerow(
                {
                    "plot_id": row["plot_id"],
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "label": row["label"],
                    "label_name": row["label_name"],
                    "text_path": row["text_path"],
                    "tsne_x": float(xy[0]),
                    "tsne_y": float(xy[1]),
                }
            )


def plot_split_panels(path, records, points):
    colors = {0: "#2878b5", 1: "#d55e00"}
    markers = {"large": "o", "paired": "^"}
    labels = {
        ("large", 0): "large/khong",
        ("large", 1): "large/co",
        ("paired", 0): "paired/khong",
        ("paired", 1): "paired/co",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    for ax, split in zip(axes, SPLITS):
        split_idx = [i for i, row in enumerate(records) if row["split"] == split]
        for dataset in ("large", "paired"):
            for label in (0, 1):
                idx = [i for i in split_idx if records[i]["dataset"] == dataset and records[i]["label"] == label]
                if not idx:
                    continue
                xy = points[idx]
                ax.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    s=18,
                    c=colors[label],
                    marker=markers[dataset],
                    alpha=0.72,
                    linewidths=0.2,
                    edgecolors="white",
                    label=labels[(dataset, label)],
                )
        ax.set_title(split)
        ax.set_xlabel("t-SNE 1")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("t-SNE 2")
    handles, legend_labels = [], []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in legend_labels:
                handles.append(handle)
                legend_labels.append(label)
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Text embedding t-SNE: large text vs paired text")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    seed_all(args.seed)
    model_name_or_path = resolve_model_source(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records, large_skipped, paired_missing, large_root, paired_root = load_records(args)
    if large_skipped:
        (out / "large_skipped_text_files.txt").write_text("\n".join(large_skipped), encoding="utf-8")
    if paired_missing:
        (out / "paired_missing_text_files.txt").write_text("\n".join(paired_missing), encoding="utf-8")
    if not records:
        raise RuntimeError("Need non-empty large/paired text records.")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = AutoModel.from_pretrained(model_name_or_path)
    requested_max_len = args.max_len or (checkpoint_max_len(args.checkpoint) if args.checkpoint else None) or 512
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

    points, used_perplexity = fit_tsne(embeddings, args.seed, args.perplexity)
    save_points(out / "text_tsne_points.csv", records, points)
    np.save(out / "text_embeddings.npy", embeddings)
    plot_split_panels(out / "text_tsne_by_split.png", records, points)

    summary = {
        "model": str(Path(model_name_or_path).resolve()) if Path(model_name_or_path).exists() else model_name_or_path,
        "large_root": str(Path(large_root).resolve()),
        "paired_root": str(Path(paired_root).resolve()),
        "max_len": max_len,
        "perplexity": used_perplexity,
        "samples": len(records),
        "by_dataset_split": {},
    }
    for dataset in ("large", "paired"):
        for split in SPLITS:
            summary["by_dataset_split"][f"{dataset}/{split}"] = sum(
                1 for row in records if row["dataset"] == dataset and row["split"] == split
            )
    with open(out / "text_tsne_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved t-SNE figure: {(out / 'text_tsne_by_split.png').resolve()}")
    print(f"Saved t-SNE points: {(out / 'text_tsne_points.csv').resolve()}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
