from datasets import Dataset, DatasetDict


def split_train_with_column(dataset: Dataset):
    # dataset should contain only train set
    assert list(dataset.keys()) == ['train'], "Dataset for splitting should contain only train set"
    train_ds = dataset['train']

    assert 'split' in train_ds.features, "No column named split which is needed for splitting"

    split_lst = train_ds['split']

    train_idx = []
    val_idx = []
    test_idx = []

    for idx, split in enumerate(split_lst):
        if 'train' in split:
            train_idx.append(idx)
        if 'val' in split:
            val_idx.append(idx)
        if 'test' in split:
            test_idx.append(idx)

    train_split = train_ds.select(indices=train_idx)
    val_split = train_ds.select(indices=val_idx)
    test_split = train_ds.select(indices=test_idx)

    return DatasetDict({"train": train_split, "validation": val_split, "test": test_split})
