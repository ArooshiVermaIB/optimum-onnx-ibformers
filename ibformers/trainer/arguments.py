import argparse
import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, TypeVar, Tuple, List

from transformers import HfArgumentParser, TrainingArguments
from transformers.hf_argparser import DataClass
from transformers import is_optuna_available


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
class ExtraModelArguments:
    """
    Arguments specific to the custom models and configs. They will be passed to model config.
    It is advised to set defaults here to None, as they will be added to every model config initialization.
    """

    bbox_scale_factor: Optional[float] = field(
        default=None, metadata={"help": "Scale factor for regression-based bbox masking task"}
    )

    smooth_loss_beta: Optional[float] = field(
        default=None, metadata={"help": "Beta parameter for regression-based bbox masking task"}
    )


@dataclass
class EnhancedTrainingArguments(TrainingArguments):
    """
    Extra training arguments passed to the training loop. Enhance to the transformers.TrainingArguments
    """

    do_hyperparam_optimization: bool = field(
        default=False,
        metadata={
            "help": "If true, the hyperparamsearch will be conducted before the training. The best parameters"
            "will be then used to train the model."
        },
    )

    hp_search_output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Output path for hyperparam optimization. Will contain search results. If not provided, output "
            "path for training will be used."
        },
    )

    hp_search_objective_name: str = field(
        default="eval_macro_f1", metadata={"help": "Metric used as an objective for hyperparam optimization."}
    )

    hp_search_do_minimize_objective: bool = field(
        default=False,
        metadata={"help": "If set, the hyperparam search will minimize the objective instead " "of maximizing it. "},
    )

    hp_search_disable_eval: bool = field(
        default=False,
        metadata={
            "help": "If set, the training runs for hyperparameter search will skip evaluation mid-training. "
            "This will save some time but will disallow optuna to perform experiment pruning. "
        },
    )

    hp_search_force_disable_early_stopping: bool = field(
        default=False,
        metadata={
            "help": "If set, the training runs for hyperparameter search will have disabled early stopping, "
            "regardless of the value of `early_stopping_patience` parameter. "
        },
    )

    hp_search_keep_trial_artifacts: bool = field(
        default=False,
        metadata={
            "help": "If set, artifacts from each trial will be kept in hp_search_output_dir. Warning: this "
            "might require a lot of free space."
        },
    )
    hp_search_log_trials_to_wandb: bool = field(
        default=True,
        metadata={
            "help": "If set, all trials will be reported to wandb. The runs will be reported in more efficient "
            "way, where only the config and the final results are logged."
        },
    )

    hp_search_param_space: Optional[List[Dict[str, Any]]] = field(
        default=None,
        metadata={"help": "Dictionary with parameter space definition for hyperparam tuning."},
    )

    hp_search_config_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to custom hyperparam space configuration file. If not provided, default will be used."},
    )

    hp_search_num_trials: int = field(
        default=50,
        metadata={"help": "Number of trials for hyperparam optimization."},
    )

    class_weights: float = field(
        default=1.0,
        metadata={"help": "Will be used to change the weight of the classes during loss computation"},
    )
    loss_type: str = field(
        default="ce_fixed_class_weights",
        metadata={"help": "Will be used to change a type of loss"},
    )
    class_weights_ins_power: float = field(
        default=0.3,
        metadata={
            "help": "Will be used to compute weithts for individual classes "
            "(1 / num_samples ** class_weights_ins_power)"
        },
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
    do_post_train_cleanup: bool = field(
        default=False, metadata={"help": "If True, the training directory will be cleaned of unnecessary files."}
    )

    def __post_init__(self):
        # superclass will overwrite this param. We want to keep the raw value to know if we should
        # disable eval after hyperparam search
        self.requested_do_eval = self.do_eval
        super().__post_init__()
        if self.class_weights > 1 and "labels" and self.label_smoothing_factor != 0.0:
            logging.warning("cannot support both label smoothing and class weighting. Label smoothing will be ignored")

        if (
            self.do_hyperparam_optimization
            and self.hp_search_do_minimize_objective
            and "loss" not in self.hp_search_objective_name
        ):
            logging.warning(
                "Requested to minimize the objective for hyperparam search, but the objecive name "
                "does not seem like a loss function. Ignore this message if this is intended. "
            )

        if self.do_hyperparam_optimization and not is_optuna_available():
            raise ImportError(
                "Hyperparameter parametrization was requested, but optuna is not installed. "
                "Please update Instabase, or install optuna manually if running locally."
            )

        if self.hp_search_param_space is not None and self.hp_search_config_file is not None:
            raise ValueError(
                "Please provide at most one of the hp_search_param_space and hp_search_config_file " "parameters."
            )


@dataclass
class DataArguments:
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
    dataset_config_json_file: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to json file with dataset config. This allows to load multiple datasets"},
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocessing_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Batch size (number of examples) to use for the preprocessing."},
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
        if (
            self.dataset_name_or_path is None
            and self.train_file is None
            and self.validation_file is None
            and self.dataset_config_json_file is None
        ):
            raise ValueError("Need either a dataset name, training/validation file or dataset config file")


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


def get_matching_commandline_params(parser: HfArgumentParser) -> Dict[str, Any]:
    """
    Extract CLI params that are compatible with the provided parser.

    Note: the CLI params does not have to contain all the required params - we just extract
    the ones that can be parsed and return them as dict.

    Args:
        parser: HfArgumentParser to use. It will be re-created to allow necessary modifications

    Returns: Dictionary of parsed hyperparams.

    """
    parser_to_modify = HfArgumentParser(parser.dataclass_types)

    for action in parser_to_modify._actions:
        action.default = argparse.SUPPRESS
        action.required = False

    parsed, _ = parser_to_modify.parse_known_args()
    return vars(parsed)
