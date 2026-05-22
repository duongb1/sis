# SIS Training Scripts

Training and evaluation code for the SIS text-rich, MRI-poor experiments.

Large data folders, model checkpoints, and generated outputs are intentionally excluded from Git. The scripts assume Kaggle input paths by default, but every entrypoint has CLI arguments.

## Entrypoints

- `train_mri.py`: MRI-only ResNet50 teacher. Saves `best_auc_model.pt`.
- `train_text.py`: large text-only PhoBERT on `texts/{train,val,test}/{co,khong}/*.txt`.
- `train_pair_text.py`: paired text-only PhoBERT on patient `.txt` files inside image folders.
- `eval_pair_text.py`: evaluate any text checkpoint directly on paired text splits.
- `train_hard_negative_reweight.py`: large-text checkpoint fine-tuning with MRI-guided hard-negative sample weights.
- `kd_mri_text.py`: MRI teacher to paired-text student KD, kept for appendix experiments.
- `train_lupi.py`: MRI-guided LUPI sample weighting, kept for appendix experiments.
- `train_dual_mri_align.py`: dual-stream auxiliary MRI feature alignment, kept for appendix experiments.
- `run_all.py`: run the synchronized pipeline and print one summary table.

## Shared Code

- `sislib/common.py`: labels, device helpers, seeding, rounding, model wrapping.
- `sislib/text_data.py`: text record collectors and datasets.
- `sislib/text_train.py`: text CE/KD train loops and evaluation.
- `sislib/mri.py`: MRI pair collectors, transforms, datasets, and ResNet factory.
- `sislib/mri_teacher.py`: MRI teacher checkpoint loading, patient-level logits, split stats, and shuffled controls.
- `sislib/metrics.py`: metrics and prediction CSV writers.

## Default Pipeline

By default, `run_all.py` runs the practical main pipeline for the hard-negative question:

1. `MRI-only teacher`: train MRI teacher on paired MRI train and test on paired test 280.
2. `Paired-only CE`: PhoBERT baseline on paired text train and paired test 280.
3. `Large-text CE`: train PhoBERT on the large text cohort.
4. `Large-text direct`: evaluate the large-text checkpoint directly on paired train/val/test, without paired fine-tuning.
5. `Large-text -> paired CE`: ordinary paired fine-tuning from the large-text checkpoint.
6. `MRI hard-neg reweight`: fine-tune from the large-text checkpoint with MRI-guided hard-negative weights.

```bash
python run_all.py \
  --images /kaggle/input/datasets/duongb/cthsis/images \
  --texts /kaggle/input/datasets/duongb/cthsis/texts \
  --out-root /kaggle/working/sis_runs
```

Optional controls and appendix runs:

```bash
python run_all.py --include-control   # add shuffled hard-negative reweighting
python run_all.py --include-ablation  # add KD/LUPI/Dual/weighted appendix runs
python run_all.py --include-all       # run main + control + appendix
```

At the end, `run_all.py` prints tables with `Acc`, `F1`, `AUC`, sensitivity, specificity, and confusion matrix `[[TN, FP], [FN, TP]]`. It also writes:

- `/kaggle/working/sis_runs/summary_results.csv`
- `/kaggle/working/sis_runs/summary_results.json`

## Hard-Negative Reweighting

The proposed method targets large-text false positives on paired hard negatives. It computes:

```text
p_text = P_text(co | text) from the large-text checkpoint
p_mri  = P_mri(co | MRI) from the MRI-only teacher
```

Default weighting on paired train:

```text
if y = 0 and p_text >= 0.7 and p_mri <= 0.3:
    weight = 3.0
elif y = 0 and p_text >= 0.7 and p_mri > 0.5:
    weight = 0.5
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

Shuffled control:

```bash
python train_hard_negative_reweight.py \
  --student /kaggle/working/sis_runs/02_large_text_ce/best_auc_phobert \
  --teacher /kaggle/working/sis_runs/00_mri_teacher/best_auc_model.pt \
  --images /kaggle/input/datasets/duongb/cthsis/images \
  --out /kaggle/working/sis_runs/06_mri_hard_negative_reweight_shuffled \
  --shuffle-teacher
```

The script saves `sample_weights.csv` with `p_text_co`, `p_mri_co`, assigned weight, and rule per train patient.

## Individual Runs

```bash
python train_mri.py --images /kaggle/input/datasets/duongb/cthsis/images
python train_text.py --data /kaggle/input/datasets/duongb/cthsis/texts
python train_pair_text.py --images /kaggle/input/datasets/duongb/cthsis/images
python eval_pair_text.py --model /kaggle/working/sis_runs/02_large_text_ce/best_auc_phobert --images /kaggle/input/datasets/duongb/cthsis/images --splits train val test
```

Appendix experiments:

```bash
python kd_mri_text.py --images /kaggle/input/datasets/duongb/cthsis/images
python train_lupi.py --images /kaggle/input/datasets/duongb/cthsis/images
python train_dual_mri_align.py --images /kaggle/input/datasets/duongb/cthsis/images
```
