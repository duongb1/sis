# SIS PhoBERT Excel Training

This repo is focused on Excel-based PhoBERT training for I63 vs non-I63 screening.

## Data

Expected Kaggle Excel root:

```text
/kaggle/input/datasets/duongbui/siscth
```

Expected files:

```text
700_co_label.xlsx
700_khong_label.xlsx
9937_co_label.xlsx
9937_khong_label.xlsx
```

Input text is built from these columns only:

```text
LYDO, HB_BENHLY, HB_BANTHAN, HB_GIADINH, KB_TOANTHAN, KB_BOPHAN
```

`STT` is ignored. `LABEL` is not used as text input.

Multiclass labels are mapped in code without changing the Excel files:

```text
Class 0: I63_INFARCTION
Class 1: OTHER_STROKE_LIKE = OTHER_CEREBROVASCULAR + STROKE_MIMIC_NEURO
Class 2: DISTANT_OTHER
```

## Protocol

Both comparison runners use the same 5-fold protocol:

```text
train/val/test = 70/10/20
split_stratify = multiclass LABEL
test is held out for final evaluation
validation is used for checkpoint selection
training updates model weights
```

Checkpoint selection:

```text
binary:    validation AUC
multitask: primary binary validation AUC
```

Threshold sweeps are reported only. They are not selected from test.

## Small Comparison

Run:

```bash
python run_small_model_compare.py
```

Models:

```text
small_binary_cls
small_multiclass_cls
small_binary_attnpool
small_multitask_cls_aux_0_5
small_multitask_attnpool_aux_0_5
```

Default output:

```text
/kaggle/working/sis_excel_5fold_default_compare_mcstrat
```

## Small Hard-Negative SupCon Comparison

Run:

```bash
python run_small_hard_supcon_compare.py
```

Models:

```text
small_binary_attnpool
small_attnpool_hard_supcon_0_1
small_attnpool_hard_supcon_0_2
small_attnpool_hard_supcon_0_3
```

The hard-negative SupCon variants use:

```text
PhoBERT -> attention pooling -> binary head
loss = binary_loss + contrastive_weight * hard_negative_supcon_loss
hard negative pair = I63_INFARCTION <-> OTHER_STROKE_LIKE
inference = binary head only
```

For reporting, compare these variants against the strongest baseline: PhoBERT with attention pooling and a binary classification head.

Default output:

```text
/kaggle/working/sis_excel_5fold_hard_supcon_mcstrat
```

## Large Comparison

Run:

```bash
python run_large_model_compare.py
```

Models:

```text
large_binary_cls
large_multiclass_cls
large_binary_attnpool
large_multitask_cls_aux_0_5
large_multitask_attnpool_aux_0_5
large_gated_mtl_aux_0_5
```

Default output:

```text
/kaggle/working/sis_excel_5fold_large_compare_mcstrat
```

## Pooling

Supported pooling modes:

```text
cls        first-token representation
attention  token-level attention pooling
gated      learnable CLS + attention fusion
```

Gated fusion:

```text
h_fused = gate * h_cls + (1 - gate) * h_attn
```

For multi-task models:

```text
primary head: non_i63 / I63_INFARCTION
aux head:     I63_INFARCTION / OTHER_STROKE_LIKE / DISTANT_OTHER
loss:         binary_loss + lambda_aux * aux_loss
inference:    primary binary head only
```

## Support Files

The comparison runners call:

```text
run_excel_5fold.py
train_text.py
sislib/
```

Each fold writes `metrics.json`, predictions, best checkpoint files, and training history. Each comparison runner writes:

```text
summary_compare.csv
```
