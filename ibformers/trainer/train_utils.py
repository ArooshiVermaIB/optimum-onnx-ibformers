import hashlib
import json
import logging
import os
from dataclasses import asdict, fields
from typing import Dict, List

import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel, Sequence
from transformers import HfArgumentParser

from ibformers.data.collators.augmenters.args import AugmenterArguments
from ibformers.data.pipelines.args import PreprocessArguments
from ibformers.trainer.arguments import (
    ModelArguments,
    DataArguments,
    EnhancedTrainingArguments,
    IbArguments,
    ExtraModelArguments,
)
from ibformers.utils import exceptions

HASH_MODULO = 1000000
MIN_DOCUMENT_SIZES = {
    "train": 5,
    "validation": 2,
    "test": 2,
}

MIN_DOCUMENT_SIZES_WITH_VAL_SPLIT = {
    "train": 4,
    "validation": 1,
    "test": 2,
}


def get_default_parser():
    """
    get all available parsers
    """
    return HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            PreprocessArguments,
            EnhancedTrainingArguments,
            IbArguments,
            AugmenterArguments,
            ExtraModelArguments,
        )
    )


def validate_dataset_sizes(raw_datasets: Dict[str, Dataset], is_eval_from_train: bool):
    min_sizes = MIN_DOCUMENT_SIZES_WITH_VAL_SPLIT if is_eval_from_train else MIN_DOCUMENT_SIZES
    for split_name, split_min_docs in min_sizes.items():
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
        if len(v) == 0 and k != "predict":
            raise ValueError(f"There is no document chosen for {k} set")
        elif len(v) == 0 and k == "predict":
            splitted_dataset[k] = train_ds.filter(lambda x: False)  # empty dataset hack
        else:
            splitted_dataset[k] = train_ds.select(indices=indices[k])
    return splitted_dataset


def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list = [l for l in label_list if l != -100]
    label_list.sort()
    return label_list


def get_class_weights_for_sc(ds: Dataset, training_args: EnhancedTrainingArguments, num_labels: int) -> Dict:
    class_weights = [1.0] * num_labels
    split_weights = [1.0] * 2
    sc_labels = ds["sc_labels"]
    if training_args.loss_type == "ce_ins":
        split_labels = [x[0] for x in sc_labels]
        split_weights = [1.0] * 2
        for split_idx, freq in zip(*np.unique(split_labels, return_counts=True)):
            split_weights[split_idx] = 1 / (freq**training_args.class_weights_ins_power)
        class_labels = [x[1] for x in sc_labels]
        class_weights = [1.0] * num_labels
        for class_idx, freq in zip(*np.unique(class_labels, return_counts=True)):
            class_weights[class_idx] = 1 / (freq**training_args.class_weights_ins_power)

    return class_weights, split_weights


def prepare_config_kwargs(ds: Dataset, training_args: EnhancedTrainingArguments) -> Dict:
    """
    Function prepares a dictionary of parameters suited to given dataset.
    Additonal kwargs will be passed to the model config for extraction or classification task
    """
    label_names = ["labels", "class_label", "token_label_ids", "tables"]
    features = ds.features
    if not any(label in features for label in label_names):
        return dict()
    for label in label_names:
        if not label in features:
            continue

        if isinstance(features[label], ClassLabel):
            label_list = features[label].names
            break
        elif isinstance(features[label], Sequence):
            if isinstance(features[label].feature, ClassLabel):
                label_list = features[label].feature.names
            else:
                logging.warning(f"Dataset labels column - {label} is not of required type - ClassLabel")
                label_list = get_label_list(ds[label])
            break
        else:
            logging.warning(f"Dataset labels column - {label} is not of required type - ClassLabel")
            label_list = get_label_list(ds[label])
            break

    label_to_id = {l: i for i, l in enumerate(label_list)}
    ib_id2label = {i: lab for lab, i in label_to_id.items()}

    if "start_positions" in features:  # for QA model
        return dict(ib_id2label=ib_id2label)
    # for Split classifier model class weights inverse proportion to the number of labels
    elif "sc_labels" in features:
        num_labels = len(label_list)
        class_weights, split_weights = get_class_weights_for_sc(ds, training_args, num_labels)
        return dict(
            num_labels=num_labels,
            label2id=label_to_id,
            id2label=ib_id2label,
            ib_id2label=ib_id2label,
            class_weights=class_weights,
            split_weights=split_weights,
        )
    elif "tables" in features:
        table_label_list = features["tables"].feature["table_label_id"].names
        if len(table_label_list) > 1:
            table_label2id = {l: i for i, l in enumerate(table_label_list)}
            table_id2label = {i: lab for lab, i in table_label2id.items()}
            return dict(
                table_id2label=table_id2label,
                table_label2id=table_label2id,
                num_labels=len(label_list),
                label2id=label_to_id,
                id2label=ib_id2label,
                ib_id2label=ib_id2label,
            )

    return dict(
        num_labels=len(label_list),
        label2id=label_to_id,
        id2label=ib_id2label,
        ib_id2label=ib_id2label,
    )


def dict_wo_nones(x):
    return {k: v for (k, v) in dict(x).items() if v is not None}


def save_pipeline_args(
    prep_args: PreprocessArguments, data_args: DataArguments, save_path: str, filename: str = "pipeline.json"
):
    """
    Save arguments required to perform an inference step
    """
    save_dict = asdict(prep_args, dict_factory=dict_wo_nones)
    save_dict["pipeline_name"] = data_args.pipeline_name
    save_dict["dataset_name_or_path"] = data_args.dataset_name_or_path
    save_dict["dataset_config_name"] = data_args.dataset_config_name

    with open(os.path.join(save_path, filename), "w", encoding="utf-8") as writer:
        json.dump(save_dict, writer, indent=2, sort_keys=True)


def force_types_into_dataclasses(*dataclass_objs):
    for obj in dataclass_objs:
        field_types = {field.name: field.type for field in fields(obj)}
        for fname, ftype in field_types.items():
            val = getattr(obj, fname)
            if ftype is int and val is not None:
                setattr(obj, fname, val)
