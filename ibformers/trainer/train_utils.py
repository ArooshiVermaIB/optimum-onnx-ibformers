from typing import Dict

from datasets import Dataset, DatasetDict, ClassLabel

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


def prepare_config_kwargs(ds: Dataset) -> Dict:
    """
    Function prepares a dictionary of parameters suited to given dataset.
    Additonal kwargs will be passed to the model config for extraction or classification task
    """
    features = ds.features
    if "labels" not in features or "start_positions" in features:
        return dict()
    if isinstance(features["labels"].feature, ClassLabel):
        label_list = features["labels"].feature.names
    else:
        raise ValueError(f"Dataset labels column is not of required type - ClassLabel")
    label_to_id = {l: i for i, l in enumerate(label_list)}
    kwargs = dict(
        num_labels=len(label_list), label_to_id=label_to_id, id2label={i: lab for lab, i in label_to_id.items()}
    )
    return kwargs
