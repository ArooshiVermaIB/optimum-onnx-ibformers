import hashlib
import logging
from typing import Dict, List

import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel

from ibformers.utils import exceptions

HASH_MODULO = 1000000
MIN_DOCUMENT_SIZES = {
    "train": 5,
    "validation": 2,
    "test": 2,
}


def validate_dataset_sizes(raw_datasets: Dict[str, Dataset]):
    for split_name, split_min_docs in MIN_DOCUMENT_SIZES.items():
        dataset_len = len(raw_datasets[split_name])
        if dataset_len < split_min_docs:
            raise exceptions.ValidationError(
                f"Dataset split {split_name} contains only {dataset_len} documents, "
                f"which is less than required minimum of {split_min_docs}."
            )


def get_split_score(doc_id: str) -> float:
    """
    Calculate split score in [0, 1] given input string.

    Args:
        doc_id: identifier used for splitting. Can be any string unique to a document.

    Returns:
        Value in [0, 1]
    """
    hex_score = hashlib.md5(doc_id.encode()).hexdigest()
    int_score = int(hex_score, 16)
    return int_score % HASH_MODULO / HASH_MODULO


def assign_split(doc_id: str, eval_size: float):
    """
    Assigns split given input doc identifier.

    Args:
        doc_id: identifier used for splitting. Can be any string unique to a document
        eval_size: percentage of dataset for validation set

    Returns:
        "train" or "validation" depending on the calculated split score.

    """
    p_score = get_split_score(doc_id)
    if p_score < eval_size:
        return "validation"
    else:
        return "train"


def split_eval_from_train_deterministic(dataset, validation_set_size: float) -> List[str]:
    """
    Generates split identifiers in a deterministic fashion.

    Each train document gets a score in [0, 1] based on its id. Each score is then assigned to a dataset
    independently of the other documents.
    The most rightful way to split a dataset. When expanding the dataset, the old documents stay
    in the same splits as before. No data leakage guaranteed.
    Non-train documents keep the same assigned split.

    However, there is a non-zero probability that all of the documents land in a single set, thus making
    the resulting split invalid.

    Args:
        dataset: Dataset with columns "id" and "split"
        validation_set_size: percentage of dataset for validation set

    Returns:
        List of "train", "validation", "test" or "predict" identifiers.
    """
    logging.info("Splitting using fully deterministic algorithm.")
    doc_id_list = dataset["id"]
    split_lst = dataset["split"]
    return [
        assign_split(doc_id, validation_set_size) if split == "train" else split
        for doc_id, split in zip(doc_id_list, split_lst)
    ]


def split_eval_from_train_semideterministic(dataset, validation_set_size: float) -> List[str]:
    """
    Generates split identifiers in a semi-deterministic fashion.

    This method first sorts all train documents by the split score, and then assigns the split using
    proper percentile.
    As opposed to `split_eval_from_train_deterministic`, the splits of some documents may change
    if the dataset is expanded with new docs.
    Non-train documents keep the same assigned split.
    Args:
        dataset: Dataset with columns "id" and "split"
        validation_set_size: percentage of dataset for validation set

    Returns:
        List of "train", "validation", "test" or "predict" identifiers.
    """
    logging.info("Splitting using semi deterministic algorithm.")
    doc_id_list = dataset["id"]
    split_lst = dataset["split"]
    all_scores = [get_split_score(doc_id) for doc_id in doc_id_list]
    train_scores = [score for score, split in zip(all_scores, split_lst) if split == "train"]
    split_value = np.quantile(train_scores, validation_set_size)
    return [
        split if split != "train" else ("validation" if score < split_value else "train")
        for split, score in zip(split_lst, all_scores)
    ]


def split_eval_from_train(dataset: Dataset, validation_set_size: float, fully_deterministic_split: bool):
    """
    Splits the train dataset into validation and train, and leaves all other documents as test.

    Args:
        dataset: Dataset with columns "id" and "split"
        validation_set_size: percentage of dataset for validation set
        fully_deterministic_split: if true, the split is made in fully deterministic fashin.

    Returns:
        DatasetDict with three splits.

    """
    if list(dataset.keys()) != ["train"]:
        raise exceptions.ValidationError(f"Dataset: {dataset} for splitting should contain only train set")

    train_ds = dataset["train"]

    if not ({"split", "id"} <= set(train_ds.features)):
        raise exceptions.ValidationError("No column `split` and `id` required for evaluation set split")

    split_fn = (
        split_eval_from_train_deterministic if fully_deterministic_split else split_eval_from_train_semideterministic
    )
    assigned_splits = split_fn(train_ds, validation_set_size)
    return create_splits_with_column(assigned_splits, train_ds)


def split_train_with_column(dataset: Dataset):
    # dataset should contain only train set
    if list(dataset.keys()) != ["train"]:
        raise exceptions.ValidationError(f"Dataset: {dataset} for splitting should contain only train set")

    train_ds = dataset["train"]

    if "split" not in train_ds.features:
        raise exceptions.ValidationError("No column named split which is needed for splitting")

    split_lst = [split if split != "test" else "validation+test" for split in train_ds["split"]]
    splitted_dataset = create_splits_with_column(split_lst, train_ds)

    return splitted_dataset


def create_splits_with_column(split_lst: List[str], train_ds: Dataset):
    indices = {"train": [], "validation": [], "test": [], "predict": []}
    for idx, split in enumerate(split_lst):
        for k, v in indices.items():
            if k in split:
                indices[k].append(idx)
    splitted_dataset = DatasetDict()
    for k, v in indices.items():
        if len(v) == 0:
            raise ValueError(f"There is no document chosen for {k} set")
        splitted_dataset[k] = train_ds.select(indices=indices[k])
    return splitted_dataset


def prepare_config_kwargs(ds: Dataset) -> Dict:
    """
    Function prepares a dictionary of parameters suited to given dataset.
    Additonal kwargs will be passed to the model config for extraction or classification task
    """
    features = ds.features
    if "labels" not in features:
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
