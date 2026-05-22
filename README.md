# SIS Training Scripts

Training and evaluation code for the SIS text-rich, MRI-poor experiments.

Large data folders, model checkpoints, and generated outputs are intentionally excluded from Git. The scripts assume Kaggle input paths by default, but every entrypoint has CLI arguments.

## Entrypoints

- `train_mri.py`: MRI-only ResNet50 teacher. Saves `best_auc_model.pt`.
- `train_text.py`: large text-only PhoBERT on `texts/{train,val,test}/{co,khong}/*.txt`.
- `train_pair_text.py`: paired text-only PhoBERT on patient `.txt` files inside image folders.
- `train_hard_negative_reweight.py`: large-text checkpoint fine-tuning with MRI-guided hard-negative sample weights.
- `run_all.py`: run the synchronized pipeline and print one summary table.

## Shared Code

- `sislib/common.py`: labels, device helpers, seeding, rounding, model wrapping.
- `sislib/text_data.py`: text record collectors and datasets.
- `sislib/text_train.py`: text CE train loop and evaluation.
- `sislib/mri.py`: MRI pair collectors, transforms, datasets, and ResNet factory.
- `sislib/mri_teacher.py`: MRI teacher checkpoint loading, patient-level logits, and split stats.
- `sislib/metrics.py`: metrics and prediction CSV writers.

## Default Pipeline

By default, `run_all.py` runs the large-text adaptation and MRI hard-negative weight sweep:

1. `Large-text -> paired CE`
2. `MRI hard-neg weight 1.25 + positive weight 1.1`
3. `MRI hard-neg weight 1.25 + positive weight 1.2`
4. `MRI hard-neg weight 1.4 + positive weight 1.2`
5. `MRI hard-neg weight 1.5 + positive weight 1.3`

`run_all.py` still trains the MRI teacher and large-text checkpoint first because those checkpoints are required dependencies.

```bash
python run_all.py \
  --images /kaggle/input/datasets/duongb/cthsis/images \
  --texts /kaggle/input/datasets/duongb/cthsis/texts \
  --out-root /kaggle/working/sis_runs
```

At the end, `run_all.py` prints one table with `Acc`, `F1`, `AUC`, sensitivity, specificity, and confusion matrix `[[TN, FP], [FN, TP]]`. It also writes:

- `/kaggle/working/sis_runs/summary_results.csv`
- `/kaggle/working/sis_runs/summary_results.json`

If a stage checkpoint already exists under `--out-root`, `run_all.py` skips that stage and reuses the existing output. Delete that stage folder if you want to train it again.

## Hard-Negative Reweighting

The proposed method targets large-text false positives on paired hard negatives. It computes:

```text
p_text = P_text(co | text) from the large-text checkpoint
p_mri  = P_mri(co | MRI) from the MRI-only teacher
```

Default weighting on paired train:

```text
if y = 0 and p_text >= 0.7 and p_mri <= 0.3:
    weight = hard_negative_weight
elif y = 1:
    weight = positive_weight
else:
    weight = 1.0
```

Direct run:

```bash
python train_hard_negative_reweight.py \
  --student /kaggle/working/sis_runs/02_large_text_ce/best_auc_phobert \
  --teacher /kaggle/working/sis_runs/00_mri_teacher/best_auc_model.pt \
  --images /kaggle/input/datasets/duongb/cthsis/images \
  --out /kaggle/working/sis_runs/05_mri_hard_negative_reweight \
  --epochs 5 \
  --lr 1e-5
```

The script saves `sample_weights.csv` with `p_text_co`, `p_mri_co`, assigned weight, and rule per train patient.

## Individual Runs

```bash
python train_mri.py --images /kaggle/input/datasets/duongb/cthsis/images
python train_text.py --data /kaggle/input/datasets/duongb/cthsis/texts
python train_pair_text.py --images /kaggle/input/datasets/duongb/cthsis/images
```
