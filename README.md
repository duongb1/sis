# SIS Stacking Pipeline

Code for paired text stacking/meta-classifier experiments without test leakage.

## Main Entrypoint

```bash
python scripts/run_stacking_meta_classifier.py \
  --paired_train_csv /path/to/paired_train.csv \
  --paired_val_csv /path/to/paired_val.csv \
  --paired_test_csv /path/to/paired_test.csv \
  --large_text_ckpt /path/to/large_text/best_auc_phobert \
  --paired_text_model_name_or_ckpt vinai/phobert-base \
  --mri_teacher_pred_csv /path/to/mri_teacher_predictions.csv \
  --output_dir /kaggle/working/stacking_run
```

`run_all.py` is a thin wrapper around the same script and accepts the same required paths.

## Required CSV Columns

Paired split CSVs:

```text
sample_id,text,label
```

`id` is also accepted as an alias for `sample_id`.

MRI teacher prediction CSV:

```text
sample_id,p_mri
```

`id,prob_co` from `train_mri.py` teacher output is also accepted.

Labels may be `0/1`, `khong/co`, or `không/có`.

## Protocol

- `paired_val` and `paired_test` are never resplit.
- OOF predictions are generated only inside `paired_train` with `StratifiedKFold`.
- Meta-classifier training uses `p_paired_oof`, not predictions from the final full-train paired model.
- `paired_val` selects weighted-average alpha, logistic-regression hyperparameters, and thresholds.
- `paired_test` is used only for final reporting.

## Reported Models

- `Paired-only CE`
- `Large-text direct`
- `Large-text -> paired CE`
- `Weighted average text-only`
- `Stacking text-only`
- `Stacking with MRI`

`Stacking with MRI` uses `p_mri` at decision time, so it is MRI-assisted late fusion. For text-only deployment, use `Weighted average text-only` or `Stacking text-only`.

## Outputs

The output directory contains:

```text
train_oof_predictions.csv
val_predictions.csv
test_predictions.csv
stacking_results.json
stacking_results.csv
best_meta_model.joblib
README_stacking_run.md
```

Metrics include accuracy, F1, AUC, sensitivity, specificity, balanced accuracy, and confusion matrix `[[TN, FP], [FN, TP]]`.

## Utility Scripts

- `train_mri.py`: train MRI-only ResNet50 teacher.
- `train_text.py`: train large text-only PhoBERT checkpoint.
- `train_pair_text.py`: train paired text-only PhoBERT baseline.
