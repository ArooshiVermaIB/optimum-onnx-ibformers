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
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
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
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

from instabase.model_training_tasks.jobs import JobMetadataClient

from ibformers.data.pipelines.pipeline import PIPELINES, prepare_dataset
from ibformers.datasets import DATASETS_PATH

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.10.0.dev0")
from ibformers.trainer.ib_utils import (
    IbCallback,
    IbArguments,
    InstabaseSDK,
    prepare_ib_params,
    HF_TOKEN,
    InstabaseSDKDummy,
    MountDetails,
    prepare_docpro_params,
)
from ibformers.trainer.train_utils import split_train_with_column
from ibformers.trainer.trainer import IbTrainer

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/token-classification/requirements.txt",
)

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
        with open(os.path.join(save_path, filename), "w", encoding="utf-8") as writer:
            json.dump(save_dict, writer, indent=2, sort_keys=True)


def run_train_doc_pro(
    hyperparams: Dict,
    dataset_paths: List[str],
    save_path: str,
    extraction_class_name: str,
    file_client: Any,
    username: str,
    job_metadata_client: Any,
    mount_details: Optional[MountDetails] = None,
    model_name: str = 'CustomModel',
    **kwargs: Any,
):
    logging.info('Starting Doc Pro Extraction Model Training ----------')
    logging.info('Arguments to this training session:')
    logging.info(f'Hyperparameters: {hyperparams}')
    logging.info(f'Dataset Paths: {dataset_paths}')
    logging.info(f'Model Name: {model_name}')
    logging.info(f'Save Path: {save_path}')
    logging.info(f'Extraction Class Name: {extraction_class_name}')

    assert hyperparams is not None
    parser = HfArgumentParser(
        (ModelArguments, DataAndPipelineArguments, TrainingArguments, IbArguments)
    )

    hparams_dict = prepare_docpro_params(
        hyperparams,
        dataset_paths,
        save_path,
        extraction_class_name,
        file_client,
        username,
        job_metadata_client,
        mount_details,
        model_name,
    )
    model_args, data_args, training_args, ib_args = parser.parse_dict(hparams_dict)

    if isinstance(ib_args.file_client, InstabaseSDKDummy):
        ibsdk = ib_args.file_client
    else:
        ibsdk = InstabaseSDK(ib_args.file_client, ib_args.username)

    callback = IbCallback(
        job_metadata_client=ib_args.job_metadata_client,
        ibsdk=ibsdk,
        username=ib_args.username,
        mount_details=ib_args.mount_details,
        model_name=ib_args.model_name,
        ib_save_path=ib_args.ib_save_path,
        upload=ib_args.upload,
    )

    run_train(
        model_args,
        data_args,
        training_args,
        ib_args,
        extra_callbacks=[callback],
        extra_load_kwargs={"ibsdk": ibsdk, "extraction_class_name": extraction_class_name},
    )


def run_train_annotator(
    hyperparams: Dict,
    dataset_filename: str,
    save_path: str,
    file_client: Any,
    username: Optional[str] = 'user',
    job_metadata_client: Optional["JobMetadataClient"] = None,
    mount_details: Optional[Dict] = None,
    model_name: Optional[str] = "CustomModel",
    **kwargs: Any,
):
    assert hyperparams is not None
    parser = HfArgumentParser(
        (ModelArguments, DataAndPipelineArguments, TrainingArguments, IbArguments)
    )

    hparams_dict = prepare_ib_params(
        hyperparams,
        dataset_filename,
        save_path,
        file_client,
        username,
        job_metadata_client,
        mount_details,
        model_name,
    )
    model_args, data_args, training_args, ib_args = parser.parse_dict(hparams_dict)

    if isinstance(ib_args.file_client, InstabaseSDKDummy):
        ibsdk = ib_args.file_client
    else:
        ibsdk = InstabaseSDK(ib_args.file_client, ib_args.username)

    callback = IbCallback(
        job_metadata_client=ib_args.job_metadata_client,
        ibsdk=ibsdk,
        username=ib_args.username,
        mount_details=ib_args.mount_details,
        model_name=ib_args.model_name,
        ib_save_path=ib_args.ib_save_path,
        upload=ib_args.upload,
    )

    run_train(
        model_args,
        data_args,
        training_args,
        ib_args,
        extra_callbacks=[callback],
        extra_load_kwargs={"ibsdk": ibsdk},
    )


def run_cmdline_train():
    parser = HfArgumentParser(
        (ModelArguments, DataAndPipelineArguments, TrainingArguments, IbArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, ib_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, ib_args = parser.parse_args_into_dataclasses()

    run_train(model_args, data_args, training_args, ib_args)


def run_train(
    model_args, data_args, training_args, ib_args, extra_callbacks=None, extra_load_kwargs=None
):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
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
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
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

    # load pipeline
    pipeline = PIPELINES[data_args.pipeline_name]
    collate_fn = pipeline["collate"]
    compute_metrics = pipeline["compute_metrics"]
    load_kwargs = pipeline["dataset_load_kwargs"]
    model_class = pipeline["model_class"]

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

    if extra_load_kwargs is not None:
        load_kwargs.update(extra_load_kwargs)

    raw_datasets = load_dataset(
        path=name_to_use,
        name=data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        data_files=data_files,
        **load_kwargs,
    )

    if 'split' in next(iter(raw_datasets.column_names.values())):
        raw_datasets = split_train_with_column(raw_datasets)

    tokenizer_name_or_path = (
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=HF_TOKEN if model_args.model_name_or_path.startswith("instabase/") else None,
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

    fn_kwargs = {
        "tokenizer": tokenizer,
        "padding": "max_length" if data_args.pad_to_max_length else False,
        "max_length": data_args.max_length,
        "chunk_overlap": data_args.chunk_overlap,
    }
    map_kwargs = {
        "num_proc": data_args.preprocessing_num_workers,
        "load_from_cache_file": not data_args.overwrite_cache,
        "fn_kwargs": fn_kwargs,
    }

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
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

    if training_args.do_train:
        features = train_dataset.features
    else:
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

    if isinstance(features["labels"].feature, ClassLabel):
        label_list = features["labels"].feature.names
    else:
        label_list = get_label_list(train_dataset["labels"])

    num_labels = len(label_list)
    label_to_id = {l: i for i, l in enumerate(label_list)}

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        label2id=label_to_id,
        id2label={i: l for l, i in label_to_id.items()},
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=HF_TOKEN if model_args.model_name_or_path.startswith("instabase/") else None,
    )

    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=HF_TOKEN if model_args.model_name_or_path.startswith("instabase/") else None,
    )

    callbacks = []
    if extra_callbacks is not None:
        callbacks.extend(extra_callbacks)

    # Data collator
    data_collator = collate_fn(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None, model=model
    )

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
        post_process_function=None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        model_save_path = os.path.join(training_args.output_dir, "model")
        trainer.save_model(model_save_path)  # Saves the tokenizer too for easy upload
        data_args.save(model_save_path)  # Saves the pipeline & data arguments
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
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
    run_cmdline_train()


if __name__ == "__main__":
    run_cmdline_train()
