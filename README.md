# SIS Training Scripts

Training and evaluation scripts for MRI-only, text-only, and knowledge-distillation experiments on the SIS dataset.

The dataset is expected to be mounted from Kaggle input paths. Large data folders, model checkpoints, and generated outputs are intentionally excluded from Git.

## Scripts

- `train_mri_classifier.py`: MRI-only ResNet50 classifier.
- `train_text_phobert_classifier.py`: large text-only PhoBERT classifier.
- `eval_text_phobert_on_image_test_patients.py`: evaluate the text-only checkpoint on image-test patient `.txt` files.
- `train_text_kd_from_mri_teacher.py`: MRI teacher to text student KD.
- `train_paired_text_kd_from_large_text_teacher.py`: large-text teacher to paired-text student KD.
