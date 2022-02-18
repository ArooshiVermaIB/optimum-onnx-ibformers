import hashlib
import json
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from transformers import EarlyStoppingCallback, is_optuna_available
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from ibformers.callbacks.wandb import ExtendedWandbCallback
from ibformers.trainer.arguments import EnhancedTrainingArguments, ModelArguments, DataAndPipelineArguments
from ibformers.trainer.hp_search.param_space import HyperParamSpace, get_default_param_config_path

if is_optuna_available():
    from optuna import Study
    from optuna.trial import FrozenTrial


OPTUNA_WANDB_PROJECT = "optuna-summary"
HP_RESULTS_SUBDIR = "hp_results"


def run_hp_search_optuna(trainer, n_trials: int, direction: str, **kwargs) -> "Study":
    """
    transformers.integrations.run_hp_search_optuna - modified to return the whole study.
    """
    import optuna

    logging_kwargs = kwargs.pop("logging_kwargs")

    def _log_study_callback(study: "Study", trial: "FrozenTrial"):
        trial_summary = get_trial_summary(trial)
        study_summary = get_study_summary(study, trainer.args)

        output_path = Path(trainer.args.output_dir) / HP_RESULTS_SUBDIR
        (output_path / f"trial-{trial.number}.json").write_text(json.dumps(trial_summary, indent=2))
        (output_path / "study_partial.json").write_text(json.dumps(study_summary, indent=2))

        if trainer.args.hp_search_log_trials_to_wandb:
            model_name = logging_kwargs["model_name"]
            task_name = logging_kwargs["task_name"]
            param_space_dict = logging_kwargs["param_space_dict"]

            short_model_name = model_name.rsplit("/", 1)[-1]
            config_hash = get_summary_hash({"output_dir": trainer.args.output_dir, "param_space": param_space_dict})
            group_name = f"{short_model_name}-{task_name}-{config_hash}"

            trial_summary_to_log = trial_summary.copy()
            params = trial_summary_to_log.pop("params")

            import wandb

            run = wandb.init(
                project=OPTUNA_WANDB_PROJECT,
                tags=[model_name, task_name],
                reinit=True,
                group=group_name,
                config={
                    "model_name": model_name,
                    "task_name": task_name,
                    "param_distributions": param_space_dict,
                    "number_of_trials": n_trials,
                    "output_dir": trainer.args.output_dir,
                    **params,
                },
                settings=wandb.Settings(_disable_stats=True),
            )
            run.log({"trial_no": trial.number, **trial_summary_to_log})
            run.finish()

    def _objective(trial, checkpoint_dir=None):
        try:
            checkpoint = None
            if checkpoint_dir:
                for subdir in os.listdir(checkpoint_dir):
                    if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                        checkpoint = os.path.join(checkpoint_dir, subdir)
            trainer.objective = None
            trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
            # If there hasn't been any evaluation during the training loop.
            if getattr(trainer, "objective", None) is None:
                metrics = trainer.evaluate()
                trainer.objective = trainer.compute_objective(metrics)
        except optuna.TrialPruned:
            raise
        finally:
            # remove the model output after each trial unless specified otherwise
            if not trainer.args.hp_search_keep_trial_artifacts:
                run_name = trainer.hp_name(trial) if trainer.hp_name is not None else f"run-{trial.number}"
                run_dir = os.path.join(trainer.args.output_dir, run_name)

                shutil.rmtree(run_dir, ignore_errors=True)

        return trainer.objective

    timeout = kwargs.pop("timeout", None)
    n_jobs = kwargs.pop("n_jobs", 1)
    study = optuna.create_study(direction=direction, **kwargs)
    study.optimize(_objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs, callbacks=[_log_study_callback])
    return study


def get_trial_summary(trial: "FrozenTrial") -> Dict[str, Any]:
    return {
        "params": trial.params,
        "objective": trial.value,
        "duration_in_seconds": trial.duration.total_seconds(),
        "state": trial.state.name,
    }


def get_study_summary(study: "Study", training_args: EnhancedTrainingArguments) -> Dict[str, Any]:
    return {
        "output_dir": str(training_args.output_dir),
        "best_params": study.best_params,
        "best_value": study.best_value,
        "number_of_trials": len(study.trials),
        "trials_summary": [get_trial_summary(trial) for trial in study.trials],
    }


def get_summary_hash(study_summary: Dict[str, Any]):
    summary_str = json.dumps(study_summary)
    config_hash = hashlib.md5(summary_str.encode()).hexdigest()
    return config_hash[:5]


def get_current_study_id_and_project() -> Optional[Tuple[str, str]]:
    import wandb

    if wandb.run is not None:
        return wandb.run.id, wandb.run.project
    return None


def reinit_previous_run(previous_run_id_and_project: Optional[Tuple[str, str]]):
    previous_run_id, previous_run_project = previous_run_id_and_project
    import wandb

    wandb.init(project=previous_run_project, id=previous_run_id, resume=True)


def create_param_space(training_args: EnhancedTrainingArguments) -> HyperParamSpace:
    if training_args.hp_search_config_file is None:
        config_path = get_default_param_config_path()
    else:
        config_path = Path(training_args.hp_search_config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Provided hp space config file {config_path} does not exist!")
    return HyperParamSpace.from_json_file(config_path)


def optimize_hyperparams(
    trainer, training_args: EnhancedTrainingArguments, model_args: ModelArguments, data_args: DataAndPipelineArguments
):
    handle_wandb_cb = "wandb" in training_args.report_to
    handle_early_stopping_cb = (
        training_args.hp_search_disable_eval or training_args.hp_search_force_disable_early_stopping
    )

    # we need to save the current run id and project to be able to resume it after optimization
    if handle_wandb_cb:
        current_run_data = get_current_study_id_and_project()
        old_wandb_callback = trainer.pop_callback(ExtendedWandbCallback)
        trainer.args.report_to = ["none"]

    param_space = create_param_space(training_args)

    old_args = deepcopy(trainer.args)
    if training_args.hp_search_disable_eval:
        trainer.args.evaluation_strategy = "no"

    if handle_early_stopping_cb:
        trainer.args.save_strategy = "no"
        early_stopping_callback = trainer.pop_callback(EarlyStoppingCallback)

    if trainer.args.hp_search_output_dir is not None:
        trainer.args.output_dir = trainer.args.hp_search_output_dir

    output_path = Path(trainer.args.output_dir) / HP_RESULTS_SUBDIR
    output_path.mkdir(exist_ok=True)
    param_space.to_json_file(output_path / "space_config.json")

    study = trainer.hyperparameter_search(
        compute_objective=lambda x: x[training_args.hp_search_objective_name],
        hp_space=param_space,
        direction="minimize" if training_args.hp_search_do_minimize_objective else "maximize",
        backend="optuna",
        n_trials=training_args.hp_search_num_trials,
        logging_kwargs={
            "param_space_dict": param_space.to_json_content(),
            "model_name": model_args.model_name_or_path,
            "task_name": data_args.task_name,
        },
    )
    study_summary = get_study_summary(study, trainer.args)

    if handle_wandb_cb:
        trainer.add_callback(old_wandb_callback)
        reinit_previous_run(current_run_data)

    if handle_early_stopping_cb:
        if early_stopping_callback is not None:
            trainer.add_callback(early_stopping_callback)

    (output_path / "study.json").write_text(json.dumps(study_summary, indent=2))

    trainer.args = old_args
    return study.best_params
