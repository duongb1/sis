# SIS Training Pipeline

Code for multi-class large text-only and small text-only training, plus direct cross-test evaluation between the two text checkpoints.

## Train From Excel Files

The training script can read `.xlsx` files directly. For the current files:

```text
700_co_label.xlsx
700_khong_label.xlsx
9937_co_label.xlsx
9937_khong_label.xlsx
```

Each row is one sample. Text input is built only from:

```text
LYDO, HB_BENHLY, HB_BANTHAN, HB_GIADINH, KB_TOANTHAN, KB_BOPHAN
```

`STT` is ignored. `LABEL` is not included in the input text.

Multi-class training uses `LABEL` as the target, with the Excel labels mapped at load time into 3 classes without modifying the Excel files:

```text
Class 0: I63_INFARCTION
Class 1: OTHER_STROKE_LIKE = OTHER_CEREBROVASCULAR + STROKE_MIMIC_NEURO
Class 2: DISTANT_OTHER
```

```bash
python train_text.py \
  --data /kaggle/input/datasets/duongb/cthsis \
  --format excel \
  --excel-task multiclass \
  --out /kaggle/working/text_excel_multiclass
```

Binary training uses `co`/`khong` inferred from each Excel filename:

```bash
python train_text.py \
  --data /kaggle/input/datasets/duongb/cthsis \
  --format excel \
  --excel-task binary \
  --out /kaggle/working/text_excel_binary
```

Multi-task training uses one shared PhoBERT encoder with a primary binary head and an auxiliary 3-class head:

```text
primary binary head: non_i63 / I63_INFARCTION
auxiliary head:      I63_INFARCTION / OTHER_STROKE_LIKE / DISTANT_OTHER
loss:                loss_binary + lambda_aux * loss_aux
```

The default auxiliary weight is `--lambda-aux 0.5`:

```bash
python train_text.py \
  --data /kaggle/input/datasets/duongb/cthsis \
  --format excel \
  --excel-task multitask \
  --lambda-aux 0.5 \
  --out /kaggle/working/text_excel_multitask
```

Excel input is split into train/val/test with stratified ratios controlled by `--val-ratio` and `--test-ratio` (both default to `0.1`).

Pooling methods:

```text
cls         Use the default first-token representation path.
attention   Learn token-level attention pooling over PhoBERT hidden states before classification.
```

Field-aware mode:

```text
concat   Join all clinical fields into one sequence.
field    Encode each clinical field separately with a shared PhoBERT encoder, then aggregate field representations with a lightweight Transformer and field attention pooling.
```

## Excel 5-Fold Protocol

Run the small-set experiments with 5 folds, where each fold uses 70% train, 10% validation, and 20% test:

```bash
python run_excel_5fold.py
```

By default this currently runs only:

```text
small_binary       /kaggle/input/datasets/duongb/cthsis/700_co_label.xlsx + /kaggle/input/datasets/duongb/cthsis/700_khong_label.xlsx, target from filename co/khong
small_multiclass   /kaggle/input/datasets/duongb/cthsis/700_co_label.xlsx + /kaggle/input/datasets/duongb/cthsis/700_khong_label.xlsx, target from mapped 3-class LABEL
small_multitask    /kaggle/input/datasets/duongb/cthsis/700_co_label.xlsx + /kaggle/input/datasets/duongb/cthsis/700_khong_label.xlsx, binary head + auxiliary 3-class head
```

Use `--only all` to also run `large_binary`, `large_multiclass`, and `large_multitask`.

Outputs are written to `/kaggle/working/sis_excel_5fold/<experiment>/fold_<0-4>/` by default. To check commands without training:

```bash
python run_excel_5fold.py --dry-run
```

To run attention pooling without overwriting the default CLS-pooling results, use a separate output directory:

```bash
python run_excel_5fold.py \
  --only small_binary \
  --pooling attention \
  --output-dir /kaggle/working/sis_excel_5fold_attnpool_binary \
  --force
```

To run field-aware binary training first:

```bash
python run_excel_5fold.py \
  --only small_binary \
  --input-mode field \
  --max-len-per-field 128 \
  --batch 8 \
  --accum 2 \
  --output-dir /kaggle/working/sis_excel_5fold_fieldaware_binary \
  --force
```

For multi-class checkpoints, `binary_i63` is evaluated by thresholding `P(I63_INFARCTION)` rather than by checking whether the 4-class argmax is `I63_INFARCTION`. The default 5-fold runner writes a binary threshold sweep for:

```text
0.30, 0.35, 0.40, 0.45, 0.50
```

After all 5 folds finish for an experiment, the runner prints mean ± std and saves:

```text
/kaggle/working/sis_excel_5fold/<experiment>/summary_5fold.json
/kaggle/working/sis_excel_5fold/<experiment>/summary_5fold.csv
```

At the end, the runner prints a compact small-model report with:

```text
5-fold summary table for small_binary, small_multiclass_to_binary, small_multitask
aggregate confusion counts for every model
FP/FN trade-off versus small_binary
best model by F1, AUC, sensitivity, specificity, and balanced accuracy
compact threshold sweep tables for small_binary and small_multiclass
```

## Train Large And Small Text Folders

For the Kaggle `sis` input folder that contains:

```text
/kaggle/input/datasets/duongb/cthsis/sis/large/train/I63_INFARCTION.csv
/kaggle/input/datasets/duongb/cthsis/sis/large/train/G45_TIA.csv
/kaggle/input/datasets/duongb/cthsis/sis/large/val/I63_INFARCTION.csv
/kaggle/input/datasets/duongb/cthsis/sis/large/test/I63_INFARCTION.csv
/kaggle/input/datasets/duongb/cthsis/sis/small/train/I63_INFARCTION.csv
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

Each CSV row is treated as one text sample. The class label is the CSV filename without `.csv`. All non-empty columns in that row are joined into the input text. `run_all.py` uses the union of labels from `large` and `small`, so cross-test checkpoints share the same class index mapping.

Metrics are reported in two ways:

```text
multi-class: all CSV filename classes
binary_i63: I63_INFARCTION = 1, every other class = 0
```

The binary metrics include a 2x2 confusion matrix in `metrics.json` under `binary_i63`.

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
