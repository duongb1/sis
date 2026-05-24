# SIS Training Pipeline

Code for MRI-only, large text-only, paired text-only training, plus direct cross-test evaluation between the two text checkpoints.

## Full Pipeline Entrypoint

Run all required stages once:

```bash
python run_all.py \
  --images /kaggle/input/datasets/duongb/cthsis/images \
  --texts /kaggle/input/datasets/duongb/cthsis/texts \
  --output_dir /kaggle/working/sis_runs
```

This runs, in order:

```text
00_mri_teacher      MRI-only ResNet50 teacher
01_large_text_ce    Large text-only PhoBERT
02_paired_text_ce   Paired text-only PhoBERT
03_cross_test       Direct checkpoint cross-tests
```

Existing outputs are skipped by default:

```text
00_mri_teacher/best_auc_model.pt
01_large_text_ce/best_auc_phobert
02_paired_text_ce/best_auc_phobert
03_cross_test/large_on_paired_test/metrics.json
03_cross_test/paired_on_large_test/metrics.json
```

Use `--force` to retrain all stages.

If the large-text checkpoint already exists elsewhere, pass:

```bash
python run_all.py \
  --images /kaggle/input/datasets/duongb/cthsis/images \
  --texts /kaggle/input/datasets/duongb/cthsis/texts \
  --large_text_ckpt /kaggle/working/sis_runs/01_large_text_ce/best_auc_phobert \
  --output_dir /kaggle/working/sis_runs
```

## Cross-Test Only

Evaluate the large text checkpoint directly on the paired test split:

```bash
python scripts/eval_text_checkpoint.py \
  --checkpoint /path/to/large_text/best_auc_phobert \
  --dataset paired \
  --images /kaggle/input/datasets/duongb/cthsis/images \
  --out /kaggle/working/sis_runs/03_cross_test/large_on_paired_test
```

Evaluate the paired text checkpoint directly on the large text test split:

```bash
python scripts/eval_text_checkpoint.py \
  --checkpoint /path/to/paired_text/best_auc_phobert \
  --dataset large \
  --texts /kaggle/input/datasets/duongb/cthsis/texts \
  --out /kaggle/working/sis_runs/03_cross_test/paired_on_large_test
```

## Reported Models

- `MRI-only teacher`
- `Large-text CE`
- `Paired-only CE`
- `Large checkpoint on paired test`
- `Paired checkpoint on large test`

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
- `scripts/eval_text_checkpoint.py`: evaluate a text checkpoint directly on either large or paired test split.
