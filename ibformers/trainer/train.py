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
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import datasets
import transformers
from datasets import DownloadConfig
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    set_seed,
    PreTrainedTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from ibformers.callbacks.early_stopping import IbEarlyStoppingCallback
from ibformers.data.collators.augmenters.args import AugmenterArguments
from ibformers.data.collators.collmenter import CollatorWithAugmentation
from ibformers.data.pipelines.args import PreprocessArguments
from ibformers.data.pipelines.pipeline import PIPELINES, prepare_dataset
from ibformers.trainer.arguments import (
    ModelArguments,
    EnhancedTrainingArguments,
    DataArguments,
    IbArguments,
    ExtraModelArguments,
    get_matching_commandline_params,
)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
from ibformers.trainer.data_loading import load_raw_dataset
from ibformers.trainer.hp_search.optimize import optimize_hyperparams

check_min_version("4.12.3")
from ibformers.trainer.train_utils import (
    force_types_into_dataclasses,
    prepare_config_kwargs,
    get_default_parser,
    save_pipeline_args,
    dict_wo_nones,
)
from ibformers.trainer import metrics_utils
from ibformers.trainer.trainer import IbTrainer

require_version(
    "datasets>=1.15.1",
    "To fix: pip install -r examples/pytorch/token-classification/requirements.txt",
)

logger = logging.getLogger(__name__)


def run_hyperparams_and_cmdline_train(hyperparams: Dict):
    parser = get_default_parser()
    parsed_cli_params = get_matching_commandline_params(parser)
    hyperparams.update(**parsed_cli_params)
    model_args, data_args, prep_args, training_args, ib_args, augmenter_args, extra_model_args = parser.parse_dict(
        hyperparams
    )

    # workaround for docpro params
    if hyperparams.get("dataset_config_name", "") in {"ib_extraction", "ib_classification", "ib_split_class"}:
        if not isinstance(data_args.train_file, list):
            data_args.train_file = [data_args.train_file]

    run_train(
        model_args,
        data_args,
        prep_args,
        training_args,
        ib_args,
        augmenter_args,
        extra_model_args,
        extra_callbacks=[],
        extra_load_kwargs={
            "extraction_class_name": data_args.extraction_class_name,
            "download_config": DownloadConfig(max_retries=3),
        },
    )


def run_cmdline_train():
    parser = get_default_parser()
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (
            model_args,
            data_args,
            prep_args,
            training_args,
            ib_args,
            augmenter_args,
            extra_model_args,
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (
            model_args,
            data_args,
            prep_args,
            training_args,
            ib_args,
            augmenter_args,
            extra_model_args,
        ) = parser.parse_args_into_dataclasses()

    run_train(model_args, data_args, prep_args, training_args, ib_args, augmenter_args, extra_model_args)


def run_train(
    model_args: ModelArguments,
    data_args: DataArguments,
    prep_args: PreprocessArguments,
    training_args: EnhancedTrainingArguments,
    ib_args: IbArguments,
    augmenter_args: AugmenterArguments,
    extra_model_args: ExtraModelArguments,
    extra_callbacks=None,
    extra_load_kwargs=None,
):
    # Explicitly cast integer type dataclass attributes, which sometimes are loaded as float
    force_types_into_dataclasses(
        model_args,
        data_args,
        prep_args,
        training_args,
        ib_args,
        augmenter_args,
        extra_model_args,
    )

    if extra_load_kwargs is None:
        extra_load_kwargs = dict()
    # Setup logging
    setup_logging(training_args)

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

    # load pipeline
    pipeline = PIPELINES[data_args.pipeline_name]
    pip_aug_kwargs = pipeline["augmenters_kwargs"]
    compute_metrics = pipeline["compute_metrics"]
    load_kwargs = pipeline["dataset_load_kwargs"]
    model_class = pipeline["model_class"]
    predict_specific_pipeline_present = "predict_preprocess" in pipeline
    if extra_load_kwargs is not None:
        load_kwargs.update(extra_load_kwargs)

    base_model_local_path, token = prepare_base_model_loading(model_args)

    raw_datasets = load_raw_dataset(data_args, load_kwargs, model_args)

    if pipeline.get("needs_tokenizer", True):
        tokenizer_name_or_path = (
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
        )

        tokenizer = AutoTokenizer.from_pretrained(
            _get_model_name_or_path(tokenizer_name_or_path, base_model_local_path),
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=token,
        )

        # Tokenizer check: this script requires a fast tokenizer.
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise ValueError(
                "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
                "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
                "requirement"
            )
    else:
        tokenizer = PreTrainedTokenizer()

    # Preprocessing the dataset
    # Padding strategy

    # get pipeline arguments which where passed (different than None)

    prep_args_to_use = asdict(prep_args, dict_factory=dict_wo_nones)
    fn_kwargs = {"tokenizer": tokenizer, **prep_args_to_use}
    map_kwargs = {
        "num_proc": data_args.preprocessing_num_workers,
        "load_from_cache_file": not data_args.overwrite_cache,
        "batch_size": data_args.preprocessing_batch_size,
    }

    if training_args.do_train or training_args.do_hyperparam_optimization:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]

        if "ibsdk" in extra_load_kwargs:
            metrics_utils.increment_document_counter(
                train_dataset,
                ib_args.job_metadata_client.job_id,
                ib_args.model_name,
                ib_args.username,
            )

        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        if data_args.shuffle_train:
            train_dataset = train_dataset.shuffle(training_args.seed)
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            fn_kwargs_for_split = {"split_name": "train", **fn_kwargs}
            map_kwargs["fn_kwargs"] = fn_kwargs_for_split
            train_dataset = prepare_dataset(train_dataset, pipeline, **map_kwargs)
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval or training_args.do_hyperparam_optimization:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        # with training_args.main_process_first(desc="validation dataset map pre-processing"):
        fn_kwargs_for_split = {"split_name": "validation", **fn_kwargs}
        map_kwargs["fn_kwargs"] = fn_kwargs_for_split
        eval_dataset = prepare_dataset(eval_dataset, pipeline, **map_kwargs)

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = raw_datasets["test"]
        if "predict" not in raw_datasets:
            logger.info("Ignoring predict dataset, as the test one is present")
            predict_raw_dataset = test_dataset.filter(lambda x: False)
        else:
            predict_raw_dataset = raw_datasets["predict"]

        if data_args.max_predict_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_predict_samples))
            predict_raw_dataset = predict_raw_dataset.select(range(data_args.max_predict_samples))

        test_ids = set(test_dataset["id"])

        if len(predict_raw_dataset) > 0:
            joint_test_predict = datasets.concatenate_datasets([test_dataset, predict_raw_dataset])
        else:
            joint_test_predict = test_dataset

        if not predict_specific_pipeline_present:
            fn_kwargs_for_split = {"split_name": "predict_test", **fn_kwargs}
            map_kwargs["fn_kwargs"] = fn_kwargs_for_split
            joint_test_predict = prepare_dataset(joint_test_predict, pipeline, **map_kwargs)

            test_dataset = joint_test_predict.filter(lambda x: x["id"] in test_ids)
            predict_dataset = joint_test_predict

        else:
            fn_kwargs_for_split = {"split_name": "test", **fn_kwargs}
            map_kwargs["fn_kwargs"] = fn_kwargs_for_split
            test_dataset = prepare_dataset(test_dataset, pipeline, **map_kwargs)
            fn_kwargs_for_split = {"split_name": "predict_test", **fn_kwargs}
            map_kwargs["fn_kwargs"] = fn_kwargs_for_split
            predict_dataset = prepare_dataset(
                joint_test_predict, pipeline, use_predict_specific_pipeline=True, **map_kwargs
            )

    config_kwargs = prepare_config_kwargs(train_dataset if training_args.do_train else eval_dataset, training_args)

    config_class = getattr(model_class, "config_class", AutoConfig)
    config = config_class.from_pretrained(
        _get_model_name_or_path(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path, base_model_local_path
        ),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=token,
        **asdict(extra_model_args, dict_factory=dict_wo_nones),
    )
    config.update(config_kwargs)

    model = model_class.from_pretrained(
        _get_model_name_or_path(model_args.model_name_or_path, base_model_local_path),
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=token,
        ignore_mismatched_sizes=True,
    )

    callbacks = []
    if extra_callbacks is not None:
        callbacks.extend(extra_callbacks)

    if training_args.early_stopping_patience > 0:
        early_stopping = IbEarlyStoppingCallback(training_args.early_stopping_patience)
        callbacks.append(early_stopping)

    # Data collator

    aug_args_passed = asdict(augmenter_args, dict_factory=dict_wo_nones)
    # not-None arguments passed as hyperparams can override pipeline specific arguments
    aug_args_with_pipeline = {**pip_aug_kwargs, **aug_args_passed}
    collator_class = pipeline.get("custom_collate", CollatorWithAugmentation)

    data_collator = collator_class(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        model=model,
        **aug_args_with_pipeline,
    )

    # Initialize our Trainer
    trainer_class = pipeline.get("trainer_class", IbTrainer)

    trainer = trainer_class(
        model=model,
        model_init=lambda x: model_class.from_pretrained(
            _get_model_name_or_path(model_args.model_name_or_path, base_model_local_path),
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=token,
            ignore_mismatched_sizes=True,
        ),
        args=training_args,
        train_dataset=train_dataset if (training_args.do_train or training_args.do_hyperparam_optimization) else None,
        eval_dataset=eval_dataset if (training_args.do_eval or training_args.do_hyperparam_optimization) else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        post_process_function=None,
    )

    # Hyperparam optimization
    if training_args.do_hyperparam_optimization:
        logger.info("*** Optimize hyperparams ***")
        study = optimize_hyperparams(trainer, training_args, model_args, data_args)
        best_params = study.best_params
        for param_name, param_value in best_params.items():
            setattr(trainer.args, param_name, param_value)
        logger.info(f"Hyperparameter optimization completed. Best hyperparams: {best_params}")

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        final_model_dir = ib_args.final_model_dir if ib_args.final_model_dir is not None else training_args.output_dir
        trainer.save_model(final_model_dir)  # Saves the tokenizer too for easy upload
        save_pipeline_args(prep_args, data_args, final_model_dir)
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        if training_args.do_post_train_cleanup:
            trainer.post_train_cleaup()

    # Evaluation
    if training_args.requested_do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(metric_key_prefix="final_eval")

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["final_eval_samples"] = min(max_eval_samples, len(eval_dataset))

        # trainer.log_metrics("eval", metrics)
        trainer.save_metrics("final_eval", metrics)

    # Evaluation
    if training_args.do_predict:
        logger.info("*** Test ***")
        trainer.test_dataset = test_dataset
        metrics = trainer.evaluate(test_dataset, metric_key_prefix="test_eval")

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(test_dataset)
        metrics["final_eval_samples"] = min(max_eval_samples, len(test_dataset))

        # trainer.log_metrics("eval", metrics)
        trainer.save_metrics("test_eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        _ = trainer.predict(predict_dataset, metric_key_prefix="predict")


def prepare_base_model_loading(model_args):
    # check if base model is already downloaded
    base_model_local_path = None
    if model_args.model_name_or_path:
        BASE_MODEL_CACHE_PREFIX = Path("/home/ibuser/.cache/instabase")
        _base_model_local_path = BASE_MODEL_CACHE_PREFIX / model_args.model_name_or_path
        logging.info(f"check if {_base_model_local_path} exists")
        if _base_model_local_path.exists():
            base_model_local_path = _base_model_local_path
            logging.info(f"from_pretrained() method will load base model locally from {base_model_local_path}")
    if model_args.model_name_or_path.startswith("instabase/"):
        # lazy import of file which is not always present in repo
        from ibformers.trainer.hf_token import HF_TOKEN

        token = HF_TOKEN
    else:
        token = None
    return base_model_local_path, token


def setup_logging(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logging.getLogger().setLevel(log_level)
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def _get_model_name_or_path(model_name_or_path: str, base_model_path: str) -> str:
    return base_model_path if base_model_path else model_name_or_path


def _mp_fn(index):
    # For xla_spawn (TPUs)
    run_cmdline_train()


if __name__ == "__main__":
    run_cmdline_train()
