from datasets import Dataset, DatasetDict

from ibformers.utils import exceptions


def split_train_with_column(dataset: Dataset):
    # dataset should contain only train set
    if list(dataset.keys()) != ["train"]:
        raise exceptions.ValidationError(f"Dataset: {dataset} for splitting should contain only train set")

    train_ds = dataset["train"]

    if "split" not in train_ds.features:
        raise exceptions.ValidationError("No column named split which is needed for splitting")

    split_lst = train_ds["split"]
    indices = {"train": [], "validation": [], "test": []}

    for idx, split in enumerate(split_lst):
        for k, v in indices.items():
            if k in split:
                indices[k].append(idx)

    splitted_dataset = DatasetDict()
    for k, v in indices.items():
        if len(v) == 0:
            raise ValueError(f"There is no document choosen for {k} set")
        splitted_dataset[k] = train_ds.select(indices=indices[k])

    return splitted_dataset
