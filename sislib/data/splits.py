from sklearn.model_selection import StratifiedKFold, train_test_split


def assign_kfold_splits(records, seed=42, n_folds=5, fold_index=0, val_ratio=0.1, split_label="target"):
    if n_folds < 2:
        raise ValueError("--n-folds must be at least 2.")
    if fold_index < 0 or fold_index >= n_folds:
        raise ValueError(f"--fold-index must be between 0 and {n_folds - 1}.")

    if split_label == "binary":
        y = [record["binary_label_name"] for record in records]
    elif split_label == "target":
        y = [record["label"] for record in records]
    else:
        raise ValueError(f"Unknown split_label: {split_label}")

    indices = list(range(len(records)))
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(splitter.split(indices, y))
    train_val_idx, test_idx = folds[fold_index]
    train_val_idx = list(train_val_idx)
    test_idx = list(test_idx)

    test_ratio = 1.0 / n_folds
    val_ratio_within_train_val = val_ratio / (1.0 - test_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_within_train_val,
        random_state=seed + fold_index,
        stratify=[y[index] for index in train_val_idx],
    )
    split_by_index = {index: "train" for index in train_idx}
    split_by_index.update({index: "val" for index in val_idx})
    split_by_index.update({index: "test" for index in test_idx})
    for index, record in enumerate(records):
        record["split"] = split_by_index[index]
        record["fold"] = fold_index
    return records
