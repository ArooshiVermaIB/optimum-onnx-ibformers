import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict

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


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataAndPipelineArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(
        default="ner", metadata={"help": "The name of the task (ner, pos...)."}
    )
    dataset_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If set, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    chunk_overlap: int = field(
        default=None,
        metadata={"help": "Overlap needed for producing multiple chunks"},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={
            "help": "Whether to return all the entity levels during evaluation or just the overall ones."
        },
    )
    pipeline_name: str = field(
        default=None,
        metadata={
            "help": "pipeline which is defining a training process. "
            "Default is None which is trying to infer it from model name"
        },
    )
    extraction_class_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Extraction class used to filter records from dataset. "
            "Only one can be used to train extraction model"
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name_or_path is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file.")

    def save(self, save_path, filename="pipeline.json"):
        save_dict = asdict(self)
        save_dict.pop('train_file')
        save_dict.pop('validation_file')
        save_dict.pop('test_file')
        with open(os.path.join(save_path, filename), "w", encoding="utf-8") as writer:
            json.dump(save_dict, writer, indent=2, sort_keys=True)


HF_TOKEN = "api_AYGJoxZMBtWlYODoAQgLKAuNVRXaGfQtjX"


@dataclass
class IbArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    username: Optional[str] = field(
        default=None, metadata={"help": "Username of person who is running the model training"}
    )
    file_client: Optional[Any] = field(
        default=None, metadata={"help": "File client object which support different file systems"}
    )
    job_metadata_client: Optional['JobMetadataClient'] = field(
        default=None,
        metadata={
            "help": "Job metadata client. Used for collecting information of training progress"
        },
    )
    mount_details: Optional[Dict] = field(
        default=None,
        metadata={"help": "Store information about S3 details"},
    )
    model_name: Optional[str] = field(
        default="CustomModel",
        metadata={
            "help": "The model name which will be appear in the model management dashboard ??"
        },
    )
    ib_save_path: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to save ib_package on the IB space"},
    )
    upload: Optional[bool] = field(
        default=None, metadata={"help": "Whether to upload model files to ib_save_path"}
    )
