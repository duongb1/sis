# SIS Training Scripts

Training and evaluation code for MRI-only, text-only, and knowledge-distillation experiments on the SIS dataset.

Large data folders, model checkpoints, and generated outputs are intentionally excluded from Git. The scripts assume Kaggle input paths by default, but every entrypoint has CLI arguments.

## Entrypoints

- `train_mri.py`: MRI-only ResNet50 classifier. Saves `best_auc_model.pt`.
- `train_text.py`: large text-only PhoBERT classifier on `texts/{train,val,test}/{co,khong}/*.txt`.
- `train_pair_text.py`: paired text-only PhoBERT classifier on patient `.txt` files inside image folders.
- `eval_img_text.py`: evaluate a text checkpoint on patient `.txt` files inside the image test folder.
- `kd_mri_text.py`: MRI teacher to paired-text student KD.
- `train_lupi.py`: large-text checkpoint to paired-text LUPI with MRI-guided CE sample weights.
- `run_all.py`: run the full synchronized pipeline once.

## Shared Code

- `sislib/common.py`: labels, device helpers, seeding, rounding, model wrapping.
- `sislib/text_data.py`: text record collectors and datasets.
- `sislib/text_train.py`: text CE/KD train loops and evaluation.
- `sislib/mri.py`: MRI pair collectors, transforms, datasets, and ResNet factory.
- `sislib/mri_teacher.py`: MRI teacher checkpoint loading, patient-level logits, split stats, and shuffled controls.
- `sislib/metrics.py`: metrics and prediction CSV writers.

## Examples

```bash
python train_text.py --data /kaggle/input/datasets/duongb/cthsis/texts
python train_pair_text.py --images /kaggle/input/datasets/duongb/cthsis/images
python eval_img_text.py --images /kaggle/input/datasets/duongb/cthsis/images
python train_mri.py --images /kaggle/input/datasets/duongb/cthsis/images
python kd_mri_text.py --images /kaggle/input/datasets/duongb/cthsis/images
python train_lupi.py --images /kaggle/input/datasets/duongb/cthsis/images
python train_lupi.py --alpha-lupi 0.0 --images /kaggle/input/datasets/duongb/cthsis/images
python run_all.py
```

## Synchronized Text Pipelines

Use the same `--student`, `--max-len`, `--epochs`, `--lr`, `--wd`, `--warmup`, `--threshold`, `--seed`, and `--accum` for the large-text to paired CE/KD/LUPI runs. The default paired protocol is `epochs=8`, `lr=2e-5`, `threshold=0.5`, best checkpoint by validation AUC.

```bash
# 1. Paired-only CE: PhoBERT -> paired train -> paired test 280
python train_pair_text.py \
  --model vinai/phobert-base \
  --images /kaggle/input/datasets/duongb/cthsis/images \
  --out /kaggle/working/paired_only_ce

# 2. Large-text CE: PhoBERT -> ~20K train -> ~20K test
python train_text.py \
  --model vinai/phobert-base \
  --data /kaggle/input/datasets/duongb/cthsis/texts \
  --out /kaggle/working/text_phobert_classifier

# 3. Large-text -> paired CE
python train_pair_text.py \
  --model /kaggle/working/text_phobert_classifier/best_auc_phobert \
  --images /kaggle/input/datasets/duongb/cthsis/images \
  --out /kaggle/working/large_to_paired_ce

# 4. Large-text -> paired MRI KD
python kd_mri_text.py \
  --student /kaggle/working/text_phobert_classifier/best_auc_phobert \
  --teacher /kaggle/working/mri_classifier/best_auc_model.pt \
  --images /kaggle/input/datasets/duongb/cthsis/images \
  --out /kaggle/working/large_to_paired_mri_kd

# 5. Large-text -> paired MRI LUPI
python train_lupi.py \
  --student /kaggle/working/text_phobert_classifier/best_auc_phobert \
  --teacher /kaggle/working/mri_classifier/best_auc_model.pt \
  --images /kaggle/input/datasets/duongb/cthsis/images \
  --out /kaggle/working/large_to_paired_lupi

# 6. Shuffled controls
python kd_mri_text.py \
  --student /kaggle/working/text_phobert_classifier/best_auc_phobert \
  --teacher /kaggle/working/mri_classifier/best_auc_model.pt \
  --images /kaggle/input/datasets/duongb/cthsis/images \
  --out /kaggle/working/large_to_paired_mri_kd_shuffled \
  --shuffle-teacher

python train_lupi.py \
  --student /kaggle/working/text_phobert_classifier/best_auc_phobert \
  --teacher /kaggle/working/mri_classifier/best_auc_model.pt \
  --images /kaggle/input/datasets/duongb/cthsis/images \
  --out /kaggle/working/large_to_paired_lupi_shuffled \
  --shuffle-teacher
```

To run everything above once with synchronized defaults:

```bash
python run_all.py \
  --images /kaggle/input/datasets/duongb/cthsis/images \
  --texts /kaggle/input/datasets/duongb/cthsis/texts \
  --out-root /kaggle/working/sis_runs
```
