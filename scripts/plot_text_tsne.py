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
    p.add_argument("--texts", default="/kaggle/input/datasets/duongb/cthsis/data/texts")
    p.add_argument("--images", default="/kaggle/input/datasets/duongb/cthsis/data/images")
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


def save_separate_points(path, rows):
    fieldnames = [
        "plot_id",
        "dataset",
        "split",
        "label",
        "label_name",
        "text_path",
        "tsne_x",
        "tsne_y",
        "perplexity",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_one_tsne(ax, rows, points, title):
    colors = {0: "#2878b5", 1: "#d55e00"}
    for label, label_name in ((0, "khong"), (1, "co")):
        idx = [i for i, row in enumerate(rows) if row["label"] == label]
        if not idx:
            continue
        xy = points[idx]
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=20,
            c=colors[label],
            alpha=0.75,
            linewidths=0.2,
            edgecolors="white",
            label=label_name,
        )
    ax.set_title(f"{title} (n={len(rows)})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.2)


def plot_separate_tsne(out, records, embeddings, seed, requested_perplexity):
    figures_dir = out / "separate_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    separate_rows = []
    summary = {}
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for row_idx, dataset in enumerate(("large", "paired")):
        for col_idx, split in enumerate(SPLITS):
            ax = axes[row_idx][col_idx]
            idx = [i for i, row in enumerate(records) if row["dataset"] == dataset and row["split"] == split]
            title = f"{dataset} / {split}"
            if len(idx) < 2:
                ax.set_title(f"{title} (n={len(idx)})")
                ax.axis("off")
                summary[f"{dataset}/{split}"] = {"samples": len(idx), "perplexity": None}
                continue

            subset_records = [records[i] for i in idx]
            subset_points, perplexity = fit_tsne(embeddings[idx], seed, requested_perplexity)
            plot_one_tsne(ax, subset_records, subset_points, title)
            summary[f"{dataset}/{split}"] = {"samples": len(idx), "perplexity": perplexity}

            single_fig, single_ax = plt.subplots(figsize=(7, 6))
            plot_one_tsne(single_ax, subset_records, subset_points, title)
            single_ax.legend(loc="best", frameon=False)
            single_fig.tight_layout()
            single_fig.savefig(figures_dir / f"text_tsne_{dataset}_{split}.png", dpi=300)
            plt.close(single_fig)

            for row, xy in zip(subset_records, subset_points):
                separate_rows.append(
                    {
                        "plot_id": row["plot_id"],
                        "dataset": row["dataset"],
                        "split": row["split"],
                        "label": row["label"],
                        "label_name": row["label_name"],
                        "text_path": row["text_path"],
                        "tsne_x": float(xy[0]),
                        "tsne_y": float(xy[1]),
                        "perplexity": float(perplexity),
                    }
                )

    handles, legend_labels = [], []
    for ax in axes.ravel():
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in legend_labels:
                handles.append(handle)
                legend_labels.append(label)
    if handles:
        fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Text embedding t-SNE by dataset and split")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out / "text_tsne_6_panels.png", dpi=300)
    plt.close(fig)
    save_separate_points(out / "text_tsne_separate_points.csv", separate_rows)
    return summary


def main():
    args = parse_args()
    seed_all(args.seed)
    model_name_or_path, model_source = resolve_model_source(args)
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

    np.save(out / "text_embeddings.npy", embeddings)
    separate_summary = plot_separate_tsne(out, records, embeddings, args.seed, args.perplexity)

    summary = {
        "model": str(Path(model_name_or_path).resolve()) if Path(model_name_or_path).exists() else model_name_or_path,
        "model_source": model_source,
        "requested_checkpoint": args.checkpoint,
        "large_root": str(Path(large_root).resolve()),
        "paired_root": str(Path(paired_root).resolve()),
        "max_len": max_len,
        "samples": len(records),
        "by_dataset_split": {},
        "separate_tsne": separate_summary,
    }
    for dataset in ("large", "paired"):
        for split in SPLITS:
            summary["by_dataset_split"][f"{dataset}/{split}"] = sum(
                1 for row in records if row["dataset"] == dataset and row["split"] == split
            )
    with open(out / "text_tsne_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved 6-panel t-SNE figure: {(out / 'text_tsne_6_panels.png').resolve()}")
    print(f"Saved separate t-SNE figures: {(out / 'separate_figures').resolve()}")
    print(f"Saved separate t-SNE points: {(out / 'text_tsne_separate_points.csv').resolve()}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
