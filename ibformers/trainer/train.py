#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.

import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path

import datasets
import numpy as np
from datasets import ClassLabel, load_dataset, concatenate_datasets

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed, )
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version
from ibformers.data.pipelines.pipeline import PIPELINES, prepare_dataset
from ibformers.datasets import DATASETS_PATH

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.10.0.dev0")
from ibformers.trainer.ib_utils import IbCallback, IbArguments, InstabaseSDK
from ibformers.trainer.trainer import IbTrainer

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
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
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
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
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If set, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    chunk_overlap: int = field(
        default=None,
        metadata={
            "help": "Overlap needed for producing multiple chunks"
        },
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
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    pipeline_name: str = field(
        default="layoutlm_sl",
        metadata={"help": "pipeline which is defining a training process"},
    )

    def __post_init__(self):
        if self.dataset_name_or_path is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                # assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                # assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


def prepare_ib_params(
    hyperparams: Dict,
    dataset_filename: str,
    save_path: str,
    file_client: Any,
    username: str,
    job_status_client: 'JobStatusClient',
    mount_details: Optional[Dict] = None,
    model_name: str = 'CustomModel'
) -> Dict:
    """
    Map parameters used by model service to names used in the Trainer
    :param hyperparams:
    :param dataset_filename:
    :param save_path:
    :param file_client:
    :param username:
    :param job_status_client:
    :param mount_details:
    :param model_name:
    :return:
    """
    out_dict = {}

    out_dict['do_train'] = True
    out_dict['do_eval'] = False
    out_dict['do_predict'] = True
    out_dict['num_train_epochs'] = hyperparams['epochs']
    out_dict['per_device_train_batch_size'] = int(hyperparams['batch_size'])
    out_dict['learning_rate'] = hyperparams['learning_rate']
    out_dict['max_grad_norm'] = hyperparams['max_grad_norm']
    out_dict['fp16'] = hyperparams['use_mixed_precision']
    out_dict['no_cuda'] = not hyperparams['use_gpu']
    out_dict['warmup_ratio'] = hyperparams['warmup']
    out_dict['weight_decay'] = hyperparams['weight_decay']
    out_dict['max_seq_length'] = int(hyperparams['chunk_size'])
    out_dict['chunk_overlap'] = int(hyperparams['stride'])
    out_dict['report_to'] = 'none'
    out_dict['logging_strategy'] = 'epoch'
    out_dict['evaluation_strategy'] = 'epoch'
    out_dict['disable_tqdm'] = True
    out_dict['logging_steps'] = 50
    if hyperparams['scheduler_type'] == "constant_schedule_with_warmup":
        out_dict['lr_scheduler_type'] = 'constant_with_warmup'
    elif hyperparams['scheduler_type'] == 'linear_schedule_with_warmup':
        out_dict['lr_scheduler_type'] = 'linear'
    else:
        out_dict['lr_scheduler_type'] = hyperparams['scheduler_type']

    out_dict['dataset_name_or_path'] = 'ibds'
    out_dict['model_name_or_path'] = hyperparams['model_name']
    if 'layoutlmv2' in hyperparams['model_name'].lower() or 'layoutxlm' in hyperparams['model_name'].lower():
        pipeline_name = 'layoutlmv2_sl'
    else:
        pipeline_name = 'layoutlm_sl'
    out_dict['pipeline_name'] = pipeline_name

    out_dict['dataset_config_name'] = 'ibds'

    out_dict['train_file'] = dataset_filename
    temp_dir = tempfile.TemporaryDirectory().name
    out_dict['output_dir'] = temp_dir
    out_dict['ib_save_path'] = save_path

    out_dict['overwrite_output_dir'] = False
    out_dict['return_entity_level_metrics'] = True

    out_dict['username'] = username
    out_dict['file_client'] = file_client
    out_dict['job_status_client'] = job_status_client
    out_dict['mount_details'] = mount_details
    out_dict['model_name'] = model_name

    return out_dict


def run_train(
    hyperparams: Optional[Dict] = None,
    dataset_filename: Optional[str] = None,
    save_path: Optional[str] = None,
    file_client: Optional[Any] = None,
    username: Optional[str] = None,
    job_status_client: Optional['JobStatusClient'] = None,
    mount_details: Optional[Dict] = None,
    model_name: Optional[str] = 'CustomModel',
    **kwargs: Any,
):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # scripts support both running from model-training-tasks and running from shell
    if hyperparams is not None:
        assert dataset_filename is not None
        assert save_path is not None
        # assert file_client is not None
        assert username is not None
        assert job_status_client is not None

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, IbArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, ib_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif hyperparams is not None:
        hparams_dict = prepare_ib_params(hyperparams, dataset_filename, save_path, file_client,
                                         username, job_status_client, mount_details, model_name)
        model_args, data_args, training_args, ib_args = parser.parse_dict(hparams_dict)
    else:
        model_args, data_args, training_args, ib_args = parser.parse_args_into_dataclasses()

    # create variable which idicate whether this is run by IB model service
    ibtrain = ib_args.file_client is not None

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # log_level = training_args.get_process_log_level()
    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file

    # Downloading and loading a dataset from the hub or from local datasets
    ds_path = Path(DATASETS_PATH) / data_args.dataset_name_or_path
    name_to_use = str(ds_path) if ds_path.is_dir() else data_args.dataset_name_or_path

    if ibtrain:
        # for debugging
        if isinstance(ib_args.file_client, InstabaseSDKDummy):
            ibsdk = ib_args.file_client
        else:
            ibsdk = InstabaseSDK(ib_args.file_client, ib_args.username)
        raw_datasets = load_dataset(
            path=name_to_use,
            name=data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            data_files=data_files,
            ibsdk=ibsdk,
        )
    else:
        raw_datasets = load_dataset(
            path=name_to_use,
            name=data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            data_files=data_files,
        )

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy

    fn_kwargs = {'tokenizer': tokenizer,
                 'padding': 'max_length' if data_args.pad_to_max_length else False,
                 'max_length': data_args.max_seq_length,
                 'chunk_overlap': data_args.chunk_overlap,
                 }
    map_kwargs = {'num_proc': data_args.preprocessing_num_workers,
                  'load_from_cache_file': not data_args.overwrite_cache,
                  'fn_kwargs': fn_kwargs
                  }
    pipeline = PIPELINES[data_args.pipeline_name]
    collate_fn = pipeline["collate"]
    compute_metrics = pipeline['compute_metrics']

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        # TODO: uncomment once transformers version will be updated
        # with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = prepare_dataset(train_dataset, pipeline, **map_kwargs)

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        # with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = prepare_dataset(eval_dataset, pipeline, **map_kwargs)

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        # with training_args.main_process_first(desc="prediction dataset map pre-processing"):
        predict_dataset = prepare_dataset(predict_dataset, pipeline, **map_kwargs)

    # Data collator
    data_collator = collate_fn(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    if training_args.do_train:
        column_names = train_dataset.column_names
        features = train_dataset.features
    else:
        column_names = eval_dataset.column_names
        features = eval_dataset.features

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features['labels'].feature, ClassLabel):
        label_list = features['labels'].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(train_dataset['labels'])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        label2id=label_to_id,
        id2label={i: l for l, i in label_to_id.items()},
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    callbacks = []
    if ibtrain:
        callbacks.append(IbCallback(job_status_client=ib_args.job_status_client,
                                    ibsdk=ibsdk,
                                    username=ib_args.username,
                                    mount_details=ib_args.mount_details,
                                    model_name=ib_args.model_name,
                                    ib_save_path=ib_args.ib_save_path,
                                    upload=ib_args.upload,
                                    ))
    #
    # def __init__(self, job_status_client: 'JobStatusClient', ibsdk: InstabaseSDK,
    #              username: str, mount_details: Dict, model_name: str, ib_save_path: str):

    # Initialize our Trainer
    trainer = IbTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        post_process_function=None
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        # TODO: remove once transforemrs lib is updated
        # if training_args.resume_from_checkpoint is not None:
        #     checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model(os.path.join(training_args.output_dir, 'model'))  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        # trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

        # Callback is saving predictions, therefore below code is not needed

        # predictions = np.argmax(predictions, axis=2)
        # # Remove ignored index (special tokens)
        # true_predictions = [
        #     [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        #     for prediction, label in zip(predictions, labels)
        # ]
        #
        # # trainer.log_metrics("predict", metrics)
        # trainer.save_metrics("predict", metrics)
        #
        # # Save predictions
        # output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
        # if trainer.is_world_process_zero():
        #     with open(output_predictions_file, "w") as writer:
        #         for prediction in true_predictions:
        #             writer.write(" ".join(prediction) + "\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    run_train()


# below is for debugging with running locally
# this code is not reached via model service as it is directly calling run_train fn
class InstabaseSDKDummy:
    def __init__(self, file_client: Any, username: str):
        # these will be ignored
        self.file_client = file_client
        self.username = username

    def ibopen(self, path: str, mode: str = 'r') -> Any:
        return open(path, mode)

    def read_file(self, file_path: str) -> str:
        with open(file_path) as f:
            return f.read()

    def write_file(self, file_path: str, content: str):
        with open(file_path, 'w') as f:
            f.write(content)


if __name__ == "__main__":

    class DummyJobStatus:
        def __init__(self):
            pass

        def update_job_status(self, task_name=None, task_data=None, task_state=None):
            if task_name:
                print(task_name)
            if task_data:
                print(task_data)
            if task_state:
                print(task_state)

    hyperparams = {
        "adam_epsilon": 1e-8,
        "batch_size": 2,
        "chunk_size": 510,
        "epochs": 5,
        "learning_rate": 5e-05,
        "loss_agg_steps": 2,
        "max_grad_norm": 1.0,
        "optimizer_type": "AdamW",
        "scheduler_type": "constant_schedule_with_warmup",
        "stride": 8,
        "use_gpu": False,
        "use_mixed_precision": False,
        "warmup": 0.0,
        "weight_decay": 0,
        "model_name": "microsoft/layoutlmv2-base-uncased"
    }
    example_dir = Path(__file__).parent.parent / "example"
    dataset_filename = os.path.join(example_dir, "UberEats.ibannotator")
    save_path = os.path.join(example_dir, "saved_model")
    sdk = InstabaseSDKDummy(None, "user")
    run_train(hyperparams, dataset_filename, save_path, sdk, 'user', DummyJobStatus())
