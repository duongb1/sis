# SIS Training Pipeline

Code for large text-only and small text-only training, plus direct cross-test evaluation between the two text checkpoints.

## Train Large And Small Text Folders

For the Kaggle `sis` input folder that contains:

```text
/kaggle/input/datasets/duongb/cthsis/sis/large/train/co.csv
/kaggle/input/datasets/duongb/cthsis/sis/large/train/khong.csv
/kaggle/input/datasets/duongb/cthsis/sis/large/val/co.csv
/kaggle/input/datasets/duongb/cthsis/sis/large/val/khong.csv
/kaggle/input/datasets/duongb/cthsis/sis/large/test/co.csv
/kaggle/input/datasets/duongb/cthsis/sis/large/test/khong.csv
/kaggle/input/datasets/duongb/cthsis/sis/small/train/co.csv
...
```

To train both folders without cross-testing, run:

```bash
python run_text_dirs.py
```

This trains:

```text
/kaggle/working/sis_runs_text_dirs/01_large_text_ce/best_auc_phobert
/kaggle/working/sis_runs_text_dirs/02_small_text_ce/best_auc_phobert
```

Each CSV row is treated as one text sample. All non-empty columns in that row are joined into the input text.

## Full Pipeline Entrypoint

Run the required large/small experiment once:

```bash
python run_all.py
```

This runs, in order:

```text
01_large_text_ce                     Train and evaluate on large
02_small_text_ce                     Train and evaluate on small
03_cross_test/large_on_small_test    Large checkpoint on small test
03_cross_test/small_on_large_test    Small checkpoint on large test
```

Existing outputs are skipped by default:

```text
01_large_text_ce/best_auc_phobert
02_small_text_ce/best_auc_phobert
03_cross_test/large_on_small_test/metrics.json
03_cross_test/small_on_large_test/metrics.json
```

Use `--force` to retrain all stages.

If a checkpoint already exists elsewhere, pass:

```bash
python run_all.py \
  --large_text_ckpt /kaggle/working/sis_runs/01_large_text_ce/best_auc_phobert \
  --small_text_ckpt /kaggle/working/sis_runs/02_small_text_ce/best_auc_phobert \
  --output_dir /kaggle/working/sis_runs
```

## Cross-Test Only

Evaluate the large text checkpoint directly on the small test split:

```bash
python scripts/eval_text_checkpoint.py \
  --checkpoint /path/to/large_text/best_auc_phobert \
  --dataset small \
  --small /kaggle/input/datasets/duongb/cthsis/sis/small \
  --out /kaggle/working/sis_runs/03_cross_test/large_on_small_test
```

Evaluate the small text checkpoint directly on the large test split:

```bash
python scripts/eval_text_checkpoint.py \
  --checkpoint /path/to/small_text/best_auc_phobert \
  --dataset large \
  --texts /kaggle/input/datasets/duongb/cthsis/sis/large \
  --out /kaggle/working/sis_runs/03_cross_test/small_on_large_test
```

## Reported Models

- `Large train/test`
- `Small train/test`
- `Large checkpoint on small test`
- `Small checkpoint on large test`

## Outputs

Each training directory contains:

```text
best_auc_model.pt or best_auc_phobert
metrics.json
training_history.csv
*_predictions_best_auc.csv
```

Each cross-test directory contains:

```text
metrics.json
test_predictions.csv
dataset_records.csv
```

Metrics include accuracy, F1, AUC, sensitivity, and specificity.

## Utility Scripts

- `train_mri.py`: train MRI-only ResNet50 teacher.
- `train_text.py`: train large text-only PhoBERT checkpoint.
- `train_pair_text.py`: train paired text-only PhoBERT baseline.
- `scripts/eval_text_checkpoint.py`: evaluate a text checkpoint directly on large, small, or paired test split.
- `scripts/plot_text_tsne.py`: extract text embeddings and plot t-SNE for large text vs paired text on train/val/test.
- `scripts/text_centroid_distance.py`: compute class-wise centroid distances between large and paired text embeddings.

## Text t-SNE

Plot both text datasets on all three splits. If the checkpoint exists, embeddings are extracted from that checkpoint;
otherwise the script falls back to the base PhoBERT model from `--model`:

```bash
python scripts/plot_text_tsne.py \
  --checkpoint /kaggle/working/sis_runs/01_large_text_ce/best_auc_phobert \
  --texts /kaggle/input/datasets/duongb/cthsis/data/texts \
  --images /kaggle/input/datasets/duongb/cthsis/data/images \
  --out /kaggle/working/sis_runs/04_text_tsne
```

Outputs:

```text
text_tsne_6_panels.png
separate_figures/text_tsne_large_train.png
separate_figures/text_tsne_large_val.png
separate_figures/text_tsne_large_test.png
separate_figures/text_tsne_paired_train.png
separate_figures/text_tsne_paired_val.png
separate_figures/text_tsne_paired_test.png
text_tsne_separate_points.csv
text_embeddings.npy
text_tsne_summary.json
```

Use `--sample-per-group N` to limit each dataset/split/label group before fitting t-SNE. To always use base PhoBERT
instead of a trained checkpoint, omit `--checkpoint` or pass `--model vinai/phobert-base`.

## Text Centroid Distance

Compute class-wise centroid distances for large vs paired text embeddings:

```bash
python scripts/text_centroid_distance.py \
  --checkpoint /kaggle/working/sis_runs/01_large_text_ce/best_auc_phobert \
  --texts /kaggle/input/datasets/duongb/cthsis/data/texts \
  --images /kaggle/input/datasets/duongb/cthsis/data/images \
  --out /kaggle/working/sis_runs/05_text_centroids
```

If `--checkpoint` is missing, the script falls back to `vinai/phobert-base`.

Outputs:

```text
centroid_distances.csv
centroid_group_counts.csv
centroid_distance_summary.json
text_centroids.npz
text_embeddings_for_centroids.npy
```
