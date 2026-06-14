# SIS PhoBERT Excel Training

This repo is focused on Excel-based PhoBERT training for I63 vs non-I63 screening.

## Code Organization

Core reusable code lives under `sislib/`:

```text
sislib/data            shared Excel text columns, label mapping, and split protocol
sislib/reports.py      5-fold aggregation, summary CSV/JSON, threshold and comparison reports
sislib/runner_utils.py subprocess runner helpers with PYTHONPATH handling
sislib/text_data.py    backward-compatible text/Excel dataset API
sislib/text_train.py   PhoBERT model classes and train/eval loops
```

Experiment scripts stay at repo root and should reuse these shared modules instead of duplicating label, split, or report logic.

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

Processed CSV input is also supported:

```text
processed_700.csv
processed_9937.csv
```

Required columns:

```text
Input_Text, Label
```

`Label=1` is mapped to `co`; `Label=0` is mapped to `khong`.

## Processed Binary Protocol

Run the requested processed CSV protocol:

```bash
python run_processed_binary_protocol.py \
  --small-csv processed_700.csv \
  --large-csv processed_9937.csv \
  --output-dir processed_binary_protocol_outputs
```

This runner trains:

```text
model_1_processed_700_5fold
  5-fold binary classification on processed_700, train/val/test = 70/10/20.

model_2_processed_9937_random
  Single random binary split on processed_9937, train/val/test = 70/10/20.
  Also evaluates the trained large model on all processed_700.

model_3_large_to_small_finetune_5fold
  Starts from the model_2 large checkpoint, then fine-tunes on each processed_700 train fold and tests the matching fold.
```

The 5-fold runs write Mean +/- Std reports to:

```text
processed_binary_protocol_outputs/model_1_processed_700_5fold/summary_5fold.csv
processed_binary_protocol_outputs/model_3_large_to_small_finetune_5fold/summary_5fold.csv
```

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

## FastText Baseline

Run the lightweight 5-fold FastText baselines for the small Excel files:

```bash
pip install fasttext
python run_fasttext_5fold.py \
  --excel-root /kaggle/input/datasets/duongbui/siscth \
  --output-dir /kaggle/working/sis_excel_5fold_fasttext_mcstrat \
  --seed 42 \
  --folds 5 \
  --val-ratio 0.1 \
  --test-ratio 0.2 \
  --threshold 0.5 \
  --thresholds 0.30,0.35,0.40,0.45,0.50
```

The runner trains four collapsed I63-vs-non-I63 baselines:

```text
fasttext_binary_all_fields
fasttext_binary_chief_exam
fasttext_multiclass_to_binary_all_fields
fasttext_multiclass_to_binary_chief_exam
```

It stratifies folds by the 3-class `LABEL` schema and writes `train.txt`, `val.txt`, `test.txt`, `model.bin`, and `metrics.json` under each model/fold directory.

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
