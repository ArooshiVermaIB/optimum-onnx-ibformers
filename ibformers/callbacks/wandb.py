import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

from transformers import TrainingArguments, TrainerState, TrainerControl
from transformers.integrations import WandbCallback

from ibformers.callbacks.utils import rewrite_logs

logger = logging.getLogger(__file__)


class ExtendedWandbCallback(WandbCallback):
    """
    Custom version of WandbCallback.

    It handles the following stuff:
    * Removing prediction data from metrics dict, as it's not displayed properly on wandb
    * Prepare table with model predictions for debugging purposes
    * Prepare table with metrics.

    In order for it tobe used with huggingface Trainer, one should subclass the trainer and
    manually replace the WandbCallback with its custom version.
    """

    def _log_summary_metrics(self, metrics_dict: Dict[str, Any]):
        if self._wandb is None:
            return
        metric_names = ["f1", "precision", "recall"]
        table_rows = defaultdict(list)

        is_classification = "test_eval_metrics" in metrics_dict
        metric_prefix = "" if is_classification else "test_eval_"
        metrics_dict = metrics_dict["test_eval_metrics"] if is_classification else metrics_dict

        for metric in metric_names:
            metric_subdict = metrics_dict[f"{metric_prefix}{metric}"]
            for dp_name, metric_value in metric_subdict.items():
                if metric_value == "NAN":
                    metric_value = None
                self._wandb.log({f"{dp_name}: {metric}": metric_value})
                table_rows[dp_name].append(metric_value)
        table = [[k] + v for k, v in table_rows.items()]
        columns = ["datapoint"] + metric_names
        self._wandb.log({"metrics": self._wandb.Table(data=table, columns=columns)})

    def _log_predictions(self, predictions):
        if self._wandb is None:
            return
        columns = [
            "Document Name",
            "Entity Name",
            "Predicted Value",
            "Gold Value",
            "Confidence",
            "Is Correct",
        ]
        prediction_data: List[List[Any]] = []
        for doc_path, doc_predictions in predictions.items():
            doc_name = Path(doc_path).name
            for entity_name, entity_values in doc_predictions["entities"].items():
                pred_value = entity_values["text"]
                gold_value = entity_values["gold_text"]
                confidence = entity_values["avg_confidence"]
                is_match = entity_values["is_match"]
                row = [
                    doc_name,
                    entity_name,
                    pred_value,
                    gold_value,
                    confidence,
                    is_match,
                ]
                prediction_data.append(row)
        self._wandb.log({"predictions": self._wandb.Table(data=prediction_data, columns=columns)})

    def setup(self, args, state, model, **kwargs):
        super().setup(args, state, model, **kwargs)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self._wandb is None:
            return
        metrics = kwargs.get("metrics").copy()  # type: ignore
        if "test_eval_predictions" in metrics:
            predictions = metrics.pop("test_eval_predictions")
            self._log_summary_metrics(metrics)
            self._log_predictions(predictions)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            logs = rewrite_logs(logs)
            self._wandb.log({**logs, "train/global_step": state.global_step})
