import argparse
import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict, TypeVar, Tuple, Union

from transformers import HfArgumentParser, TrainingArguments
from transformers.hf_argparser import DataClass


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
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class EnhancedTrainingArguments(TrainingArguments):
    """
    Extra training arguments passed to the training loop. Enhance to the transformers.TrainingArguments
    """

    class_weights: float = field(
        default=1.0,
        metadata={"help": "Will be used to change the weight of the classes during loss computation"},
    )
    early_stopping_patience: int = field(
        default=0,
        metadata={
            "help": "The number of epochs to wait before stopping if there is no improvement on validation "
            "set. 0 means disabled early stopping."
        },
    )
    max_no_annotation_examples_share: float = field(
        default=1.0,
        metadata={"help": "Will limit amount of chunks with no labels inside"},
    )

    bbox_scale_factor: float = field(
        default=500.0, metadata={"help": "Scale factor for regression-based bbox masking task"}
    )

    smooth_loss_beta: float = field(
        default=1.0, metadata={"help": "Beta parameter for regression-based bbox masking task"}
    )

    def __post_init__(self):
        super().__post_init__()
        if self.class_weights > 1 and "labels" and self.label_smoothing_factor != 0.0:
            logging.warning("cannot support both label smoothing and class weighting. Label smoothing will be ignored")


@dataclass
class DataAndPipelineArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
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
        default=64,
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
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
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
    validation_set_size: float = field(
        default=0.0,
        metadata={
            "help": "The size of the validation size as a fraction of train set. If 0, no validation set is "
            "created, and the test set serves as validation. There is no early stopping available "
            "in that case"
        },
    )
    fully_deterministic_eval_split: bool = field(
        default=False,
        metadata={
            "help": "If True, the splits will be created in a fully deterministic fashion, i.e. "
            "each document id will always end up in the same split. This might result in "
            "invalid splits for small datasets. If False, the specified set size will be "
            "selected."
        },
    )

    def __post_init__(self):
        if self.dataset_name_or_path is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")

    def save(self, save_path, filename="pipeline.json"):
        save_dict = asdict(self)
        save_dict.pop("train_file")
        save_dict.pop("validation_file")
        save_dict.pop("test_file")
        with open(os.path.join(save_path, filename), "w", encoding="utf-8") as writer:
            json.dump(save_dict, writer, indent=2, sort_keys=True)


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
    job_metadata_client: Optional["JobMetadataClient"] = field(
        default=None,
        metadata={"help": "Job metadata client. Used for collecting information of training progress"},
    )
    mount_details: Optional[Dict] = field(
        default=None,
        metadata={"help": "Store information about S3 details"},
    )
    model_name: Optional[str] = field(
        default="CustomModel",
        metadata={"help": "The model name which will be appear in the model management dashboard ??"},
    )
    ib_save_path: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to save ib_package on the IB space"},
    )
    upload: Optional[bool] = field(default=None, metadata={"help": "Whether to upload model files to ib_save_path"})
    final_model_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to save final model, it can be different location than checkpoint files"},
    )


T = TypeVar("T", bound=Tuple[DataClass, ...])


def update_params_with_commandline(param_dataclasses: T) -> T:
    """
    Update the values of input dataclasses with the values passed in CLI.

    Useful if the user want to overwrite base parameters from configuration with custom values.


    Args:
        param_dataclasses: Tuple of dataclasses to update

    Returns:
        Tuple of updated dataclasses, in the same order as the input ones.

    """
    outputs = []

    for param_dataclass in param_dataclasses:
        parser = HfArgumentParser(type(param_dataclass))

        for action in parser._actions:
            action.default = argparse.SUPPRESS
            action.required = False

        parsed, _ = parser.parse_known_args()
        new_params = deepcopy(param_dataclass)
        for key, value in vars(parsed).items():
            setattr(new_params, key, value)

        outputs.append(new_params)
    return (*outputs,)
