import json
import logging
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import datasets
from datasets import load_dataset, DatasetDict, Dataset
from datasets.data_files import DataFilesDict
from typing import Dict, Any, Optional, Union, Sequence

from ibformers.datasets import DATASETS_PATH
from ibformers.trainer.arguments import DataAndPipelineArguments, ModelArguments
from ibformers.trainer.train_utils import split_eval_from_train, split_train_with_column, validate_dataset_sizes

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    dataset_name_or_path: str
    dataset_config_name: str
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None


def load_raw_dataset(
    data_args: DataAndPipelineArguments, load_kwargs: Dict[str, Any], model_args: ModelArguments
) -> DatasetDict:
    if data_args.dataset_config_json_file is not None:
        logger.info(f"Loading dataset from {data_args.dataset_config_json_file} config.")
        return load_raw_datasets_from_config(data_args, load_kwargs, model_args)
    logger.info("Loading dataset from args.")
    return load_raw_dataset_from_args(data_args, load_kwargs, model_args)


def load_raw_dataset_from_args(
    data_args: DataAndPipelineArguments, load_kwargs: Dict[str, Any], model_args: ModelArguments
) -> DatasetDict:
    data_files = _prepare_data_files(data_args)
    # Downloading and loading a dataset from the hub or from local datasets
    raw_datasets = _load_raw_dataset(
        data_files, data_args.dataset_name_or_path, data_args.dataset_config_name, data_args, load_kwargs, model_args
    )
    return raw_datasets


def load_raw_datasets_from_config(
    data_args: DataAndPipelineArguments, load_kwargs: Dict[str, Any], model_args: ModelArguments
) -> DatasetDict:
    config_path = Path(data_args.dataset_config_json_file)
    dataset_configs = [DatasetConfig(**kwargs) for kwargs in json.loads(config_path.read_text())]
    raw_datasets = [
        _load_dataset_from_config(
            dataset_config,
            data_args,
            load_kwargs,
            model_args,
        )
        for dataset_config in dataset_configs
    ]
    joined_datasets = DatasetDict()
    for key in ["train", "validation", "test", "predict"]:
        datasets_to_use = [d[key] for d in raw_datasets if key in d]
        if len(datasets_to_use) > 0:
            joined_datasets[key] = concatenate_datasets(datasets_to_use)
    return joined_datasets


def _load_dataset_from_config(
    dataset_config: DatasetConfig,
    data_args: DataAndPipelineArguments,
    load_kwargs: Dict[str, Any],
    model_args: ModelArguments,
):
    data_files = _prepare_data_files(dataset_config)
    raw_datasets = _load_raw_dataset(
        data_files,
        dataset_config.dataset_name_or_path,
        dataset_config.dataset_config_name,
        data_args,
        load_kwargs,
        model_args,
    )
    return raw_datasets


def concatenate_datasets(datasets_to_concat: Sequence[Dataset]) -> Dataset:
    columns = [d.column_names for d in datasets_to_concat]
    common_columns = reduce(lambda x, y: set(x).intersection(set(y)), columns)
    datasets_with_matched_cols = [
        d.remove_columns([c for c in d.column_names if c not in common_columns]) for d in datasets_to_concat
    ]
    return datasets.concatenate_datasets(datasets_with_matched_cols)


def _load_raw_dataset(
    data_files: DataFilesDict,
    dataset_name_or_path: str,
    dataset_config_name: str,
    data_args: DataAndPipelineArguments,
    load_kwargs: Dict[str, Any],
    model_args: ModelArguments,
):
    ds_path = Path(DATASETS_PATH) / dataset_name_or_path
    name_to_use = str(ds_path) if ds_path.is_dir() else dataset_name_or_path
    raw_datasets = load_dataset(
        path=name_to_use,
        name=dataset_config_name,
        cache_dir=model_args.cache_dir,
        data_files=data_files,
        **load_kwargs,
    )
    # workaround currently only for docpro dataset which require loading into single dataset as information about split
    # could be obtained after loading a record
    is_docpro_training = "split" in next(iter(raw_datasets.column_names.values()))
    if is_docpro_training and data_args.validation_set_size > 0:
        raw_datasets = split_eval_from_train(
            raw_datasets, data_args.validation_set_size, data_args.fully_deterministic_eval_split
        )
        validate_dataset_sizes(raw_datasets, is_eval_from_train=True)
    elif is_docpro_training:
        raw_datasets = split_train_with_column(raw_datasets)
        validate_dataset_sizes(raw_datasets, is_eval_from_train=False)
    for key, dataset in raw_datasets.items():
        logger.warning(f"Dataset: {key} has {len(dataset)} examples.")
    return raw_datasets


def _prepare_data_files(data_args: Union[DatasetConfig, DataAndPipelineArguments]) -> DataFilesDict:
    data_files = DataFilesDict()
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
    return data_files
