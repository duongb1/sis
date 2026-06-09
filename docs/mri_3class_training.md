# MRI 3-Class Training

Cleaned MRI data can be represented as one flat case directory:

```text
/kaggle/input/datasets/duongbui/siscth/mri/images/{STT}/ADC/*.JPG
/kaggle/input/datasets/duongbui/siscth/mri/images/{STT}/DWI/*.JPG
```

The accompanying metadata and split files are:

```text
/kaggle/input/datasets/duongbui/siscth/mri/mri_3class.xlsx
/kaggle/input/datasets/duongbui/siscth/mri/mri_3class_folds.csv
```

MRI labels are derived from the major `MAICD` code:

```text
I63_INFARCTION      = I63
OTHER_STROKE_LIKE  = I60-I62, I64-I69, G45, G03, G04, G40, H81, R42, R51, S06, D43, Q28
DISTANT_OTHER      = all remaining major ICD groups
```

Generate the training manifest and group-aware fold file:
The checked-in defaults already point at the Kaggle dataset path:

```bash
python scripts/build_mri_3class_manifest.py
```

Train one fold:

```bash
python train_mri_3class.py \
  --fold-index 0 \
  --epochs 10 \
  --batch 4 \
  --max-images-per-case 16
```

Run all five folds:

```bash
python run_mri_3class_5fold.py \
  --epochs 10 \
  --batch 4 \
  --max-images-per-case 16
```

Each case is trained as a bag of paired MRI slices. Every model input is a 3-channel tensor where channel 0 is ADC, channel 1 is DWI, and channel 2 is zeros: `[ADC, DWI, 0]`. A ResNet-50 encoder processes the paired slices and mean-pools image features into one case-level prediction. Metrics include native 3-class accuracy, macro-F1, weighted-F1, balanced accuracy, per-class metrics, confusion matrix, and collapsed `I63` vs `non-I63` metrics.
