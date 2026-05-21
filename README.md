# SIS Training Scripts

Training and evaluation code for MRI-only, text-only, and knowledge-distillation experiments on the SIS dataset.

Large data folders, model checkpoints, and generated outputs are intentionally excluded from Git. The scripts assume Kaggle input paths by default, but every entrypoint has CLI arguments.

## Entrypoints

- `train_mri.py`: MRI-only ResNet50 classifier. Saves `best_auc_model.pt`.
- `train_text.py`: large text-only PhoBERT classifier on `texts/{train,val,test}/{co,khong}/*.txt`.
- `eval_img_text.py`: evaluate a text checkpoint on patient `.txt` files inside the image test folder.
- `kd_mri_text.py`: MRI teacher to paired-text student KD.
- `kd_text_pair.py`: large-text teacher to paired-text student KD.

## Shared Code

- `sislib/common.py`: labels, device helpers, seeding, rounding, model wrapping.
- `sislib/text_data.py`: text record collectors and datasets.
- `sislib/text_train.py`: text CE/KD train loops and evaluation.
- `sislib/mri.py`: MRI pair collectors, transforms, datasets, and ResNet factory.
- `sislib/metrics.py`: metrics and prediction CSV writers.

## Examples

```bash
python train_text.py --data /kaggle/input/datasets/duongb/cthsis/texts
python eval_img_text.py --images /kaggle/input/datasets/duongb/cthsis/images
python train_mri.py --images /kaggle/input/datasets/duongb/cthsis/images
python kd_text_pair.py --images /kaggle/input/datasets/duongb/cthsis/images
python kd_mri_text.py --images /kaggle/input/datasets/duongb/cthsis/images
```
