# MRI 3-Class Training

Cleaned MRI data can be represented as one flat case directory:

```text
images_3class/{STT}/ADC/*.JPG
images_3class/{STT}/DWI/*.JPG
```

The accompanying metadata file is:

```text
mri_3class.xlsx
```

MRI labels are derived from the major `MAICD` code:

```text
I63_INFARCTION      = I63
OTHER_STROKE_LIKE  = I60-I62, I64-I69, G45, G03, G04, G40, H81, R42, R51, S06, D43, Q28
DISTANT_OTHER      = all remaining major ICD groups
```

Generate the training manifest and group-aware fold file:

```bash
python scripts/build_mri_3class_manifest.py \
  --image-root images_3class \
  --co-excel 700_co.xlsx \
  --khong-excel 700_khong.xlsx \
  --out-manifest mri_3class_manifest.csv \
  --out-folds mri_3class_folds.csv
```

Train one fold:

```bash
python train_mri_3class.py \
  --folds-csv mri_3class_folds.csv \
  --fold-index 0 \
  --out outputs/mri_3class/fold_0 \
  --epochs 10 \
  --batch 4 \
  --max-images-per-case 16
```

Run all five folds:

```bash
python run_mri_3class_5fold.py \
  --folds-csv mri_3class_folds.csv \
  --output-dir outputs/mri_3class_5fold \
  --epochs 10 \
  --batch 4 \
  --max-images-per-case 16
```

Each case is trained as a bag of MRI slices. The CNN encodes selected ADC/DWI images and mean-pools image features into one case-level prediction. Metrics include native 3-class accuracy, macro-F1, weighted-F1, balanced accuracy, per-class metrics, confusion matrix, and collapsed `I63` vs `non-I63` metrics.
