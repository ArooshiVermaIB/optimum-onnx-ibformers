from __future__ import absolute_import

import logging
import tempfile
from argparse import Namespace
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Process
from pathlib import Path
from typing import List

import wandb
from transformers import HfArgumentParser

from ibformers.trainer.train import run_hyperparams_and_cmdline_train
from .config import BENCHMARKS_REGISTRY, MODEL_PARAMS_REGISTRY

logger = logging.getLogger(__name__)

WANDB_DEFAULT_PROJECT = "benchmark-test"
WANDB_DEFAULT_ENTITY = "instabase"


@dataclass
class BenchmarkArguments:
    models_to_run: List[str] = field(
        default_factory=lambda: MODEL_PARAMS_REGISTRY.available_configs,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    output_path: Path = field(default=None, metadata={"help": "Path where the benchmark results will be stored to."})
    benchmarks_to_run: List[str] = field(
        default_factory=lambda: BENCHMARKS_REGISTRY.available_configs,
        metadata={f"help": f"Name of challenge to run. Available: f{BENCHMARKS_REGISTRY.available_configs}"},
    )
    run_in_subprocess: bool = field(default=False)
    model_config: str = field(
        default=None,
        metadata={
            "help": f"Huggingface-compatible identifier of base model for `models_to_run`. The params will be "
            f"loaded from selected model config entry."
        },
    )
    extra_tags: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": f"Extra wandb tags for all jobs in this run."},
    )
    disable_wandb: bool = field(default=False, metadata={"help": f"If set, disables all wandb logging for this run."})
    wandb_project: str = field(
        default=WANDB_DEFAULT_PROJECT, metadata={"help": f"Default wandb project to log train/eval metrics to"}
    )
    wandb_entity: str = field(
        default=WANDB_DEFAULT_ENTITY, metadata={"help": f"Default wandb entity to log train/eval metrics to"}
    )


def configure_wandb(model_name_or_path: str, benchmark_id: str, benchmark_args: BenchmarkArguments):
    wandb.ensure_configured()  # TODO: does it actually check if wandb works properly?

    wandb.init(
        project=benchmark_args.wandb_project,
        entity=benchmark_args.wandb_entity,
        tags=["/".join(model_name_or_path.split("/")[-2:]), benchmark_id] + benchmark_args.extra_tags,
        reinit=True,
    )

    wandb.log({"task_name": benchmark_id})


def run_single_benchmark(benchmark_id: str, model_name_or_path: str, benchmark_args: BenchmarkArguments):
    supress_errors = benchmark_args.run_in_subprocess
    try:

        benchmark_config = BENCHMARKS_REGISTRY.get_config(benchmark_id)
        model_config_name = (
            benchmark_args.model_config if benchmark_args.model_config is not None else model_name_or_path
        )
        model_config = MODEL_PARAMS_REGISTRY.get_config(model_config_name)

        hyperparams = model_config.hyperparams.copy()
        hyperparams["model_name_or_path"] = model_name_or_path
        if benchmark_args.disable_wandb:
            hyperparams["report_to"] = "none"
        else:
            hyperparams["report_to"] = "wandb"
            configure_wandb(model_name_or_path, benchmark_id, benchmark_args)

        now_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        model_name_or_id_for_output = model_name_or_path.replace("/", "_")
        output_path = benchmark_args.output_path / benchmark_id / f"{model_name_or_id_for_output}_{now_str}"

        dataset_hyperparams = benchmark_config.hyperparams.copy()
        hyperparams.update(**dataset_hyperparams)
        hyperparams["output_dir"] = output_path
        hyperparams["task_name"] = benchmark_id

        output_path.mkdir(exist_ok=True, parents=True)

        run_hyperparams_and_cmdline_train(hyperparams)
    except Exception as e:
        logger.error(
            f"Encountered exception when running benchmark {benchmark_id} for model {model_name_or_path}."
            f"Full error: {e}"
        )
        if not supress_errors:
            raise e
    finally:
        wandb.finish()


def run_benchmark_in_subprocess(benchmark_id: str, model_name_or_path: str, benchmark_args: BenchmarkArguments):
    p = Process(target=run_single_benchmark, args=(benchmark_id, model_name_or_path, benchmark_args))
    p.start()
    p.join()
    p.close()


def run_benchmarks_for_single_model(selected_model: str, benchmark_args: BenchmarkArguments):
    # all_benchmarks = BenchmarkRegistry.available_benchmarks
    logger.info(f"Running benchmarks for model {selected_model}")
    target_fn = run_benchmark_in_subprocess if benchmark_args.run_in_subprocess else run_single_benchmark
    for selected_benchmark in benchmark_args.benchmarks_to_run:
        logger.info(f"Running benchmark {selected_benchmark} for model {selected_model}")
        target_fn(selected_benchmark, selected_model, benchmark_args)


def run_benchmarks(benchmark_args: BenchmarkArguments):
    logger.info(f"Selected models: {benchmark_args.models_to_run}")
    logger.info(f"Selected benchmarks: {benchmark_args.benchmarks_to_run}")
    for selected_model in benchmark_args.models_to_run:
        # model_name_or_id_for_output = model_name_or_id.replace("/", "_")
        run_benchmarks_for_single_model(selected_model, benchmark_args)


def _validate_params(params: Namespace):
    unknown_models = [m for m in params.models_to_run if m not in MODEL_PARAMS_REGISTRY.available_configs]
    unknown_benchmarks = [b for b in params.benchmarks_to_run if b not in BENCHMARKS_REGISTRY.available_configs]

    if len(unknown_models) == 0:
        logging.warning(
            f"Found models without an entry in model param config: {unknown_models}. They "
            f"will be executed with the default parameters. "
        )

    assert len(unknown_benchmarks) == 0, f"Unknown benchmarks: {unknown_models}, {unknown_benchmarks}"


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    parser = HfArgumentParser(BenchmarkArguments)
    benchmark_args, _ = parser.parse_known_args()
    benchmark_args = BenchmarkArguments(**vars(benchmark_args))
    _validate_params(benchmark_args)
    if benchmark_args.output_path is None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            benchmark_args.output_path = Path(tmp_dir)
            run_benchmarks(benchmark_args)
    else:
        run_benchmarks(benchmark_args)
