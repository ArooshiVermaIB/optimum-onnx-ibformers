from __future__ import absolute_import

import logging
import tempfile
from argparse import Namespace
from dataclasses import dataclass, field
from multiprocessing import Process
from pathlib import Path
from typing import List, Optional

import wandb
from transformers import HfArgumentParser

from ibformers.trainer.train import run_hyperparams_and_cmdline_train
from .config import BENCHMARKS_REGISTRY, MODEL_PARAMS_REGISTRY

logger = logging.getLogger(__name__)

WANDB_DEFAULT_PROJECT = "benchmark-test"
WANDB_DEFAULT_ENTITY = "instabase"


def configure_wandb(model_name_or_path: str, benchmark_id: str):
    wandb.ensure_configured()  # TODO: does it actually check if wandb works properly?

    wandb.init(
        project=WANDB_DEFAULT_PROJECT, entity=WANDB_DEFAULT_ENTITY, tags=[model_name_or_path, benchmark_id], reinit=True
    )

    wandb.log({"task_name": benchmark_id})


def run_single_benchmark(
    benchmark_id: str,
    model_name_or_path: str,
    output_path: Path,
    model_config_name: Optional[str] = None,
    supress_errors: bool = False,
):
    try:
        configure_wandb(model_name_or_path, benchmark_id)

        benchmark_config = BENCHMARKS_REGISTRY.get_config(benchmark_id)
        model_config_name = model_config_name if model_config_name is not None else model_name_or_path
        model_config = MODEL_PARAMS_REGISTRY.get_config(model_config_name)

        hyperparams = model_config.hyperparams.copy()
        hyperparams["report_to"] = "wandb"
        hyperparams["model_name_or_path"] = model_name_or_path

        dataset_hyperparams = benchmark_config.hyperparams.copy()
        hyperparams.update(**dataset_hyperparams)
        hyperparams["output_dir"] = output_path

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


def run_benchmark_in_subprocess(
    benchmark_id: str, model_name_or_path: str, output_path: Path, model_config_name: Optional[str]
):
    p = Process(
        target=run_single_benchmark, args=(benchmark_id, model_name_or_path, output_path, model_config_name, True)
    )
    p.start()
    p.join()
    p.close()


def run_benchmarks_for_single_model(
    model_name_or_path: str,
    output_path: Path,
    benchmarks_to_run: List[str],
    model_config_name: Optional[str],
    run_in_subprocess: bool,
):
    # all_benchmarks = BenchmarkRegistry.available_benchmarks
    logger.info(f"Running benchmarks for model {model_name_or_path}")
    target_fn = run_benchmark_in_subprocess if run_in_subprocess else run_single_benchmark
    for benchmark_id in benchmarks_to_run:
        logger.info(f"Running benchmark {benchmark_id} for model {model_name_or_path}")
        target_fn(benchmark_id, model_name_or_path, output_path / benchmark_id, model_config_name)


def run_benchmarks(
    models_to_run: List[str],
    output_path: Path,
    benchmarks_to_run: List[str],
    model_config_name: Optional[str],
    run_in_subprocess: bool,
):
    logger.info(f"Selected models: {models_to_run}")
    logger.info(f"Selected benchmarks: {benchmarks_to_run}")
    for model_name_or_id in models_to_run:
        model_name_or_id_for_output = model_name_or_id.replace("/", "_")
        run_benchmarks_for_single_model(
            model_name_or_id,
            output_path / model_name_or_id_for_output,
            benchmarks_to_run,
            model_config_name,
            run_in_subprocess,
        )


@dataclass
class BenchmarkArguments:
    models_to_run: List[str] = field(
        default_factory=lambda: MODEL_PARAMS_REGISTRY.available_configs,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    output_path: Path = field(default=None, metadata={"help": "Path where the benchmark results will be stored to."})
    benchmark_to_run: List[str] = field(
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


def _validate_params(params: Namespace):
    unknown_models = [m for m in params.models_to_run if m not in MODEL_PARAMS_REGISTRY.available_configs]
    unknown_benchmarks = [b for b in params.benchmark_to_run if b not in BENCHMARKS_REGISTRY.available_configs]

    assert (
        len(unknown_models) == 0 and len(unknown_benchmarks) == 0
    ), f"Unknown models or/and benchmarks: {unknown_models}, {unknown_benchmarks}"


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    parser = HfArgumentParser(BenchmarkArguments)
    benchmark_args, _ = parser.parse_known_args()
    # _validate_params(benchmark_args)
    if benchmark_args.output_path is None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_benchmarks(
                benchmark_args.models_to_run,
                Path(tmp_dir),
                benchmark_args.benchmark_to_run,
                benchmark_args.model_config,
                benchmark_args.run_in_subprocess,
            )
    else:
        run_benchmarks(
            benchmark_args.models_to_run,
            benchmark_args.output_path,
            benchmark_args.benchmark_to_run,
            benchmark_args.model_config,
            benchmark_args.run_in_subprocess,
        )
