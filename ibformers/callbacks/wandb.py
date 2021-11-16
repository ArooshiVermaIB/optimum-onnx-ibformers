import logging
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional, List

from transformers import TrainingArguments, TrainerState, TrainerControl
from transformers.integrations import WandbCallback

logger = logging.getLogger(__file__)
VALID_PREFIXES = ['train', 'eval', 'final_eval']


def _is_metric(logs: Dict[str, Any]) -> bool:
    for key in logs.keys():
        if key.endswith('_predictions'):
            return True
    return False


def prepare_metrics(metrics_dict: Dict[str, Any], prefix: str) -> None:
    """
    Prepare metrics dictionary for logging.

    By default, everything is logged to wandb. This includes model predictions in their unstructured
    form.
    """
    metrics_dict.pop(f'{prefix}_predictions')


def get_key_prefix(logs: Dict[str, Any]) -> Optional[str]:
    """
    Extract the prefix from metric dictionary.

    The prefixes are assumed to be in `VALID_PREFIXES` constant.
    Args:
        logs: dictionary of logs

    Returns:
        Prefix string, if found. `None` otherwise
    """
    log_items = logs.keys()
    for prefix in VALID_PREFIXES:
        for key in log_items:
            if key.startswith(prefix):
                return prefix
    logging.debug(f'No prefix was found for logs with keys {log_items}!')
    return None


def rename_keys(logs_dict: Dict, prefix: str) -> None:
    """
    Rename keys for grouping into panels in wandb.

    If the provided prefix is found in the dictionary key, it is removed and placed as a parent "directory".
    If there is no prefix in key, it is placed in front of the metric name.
    Args:
        logs_dict: dictionary of logs
        prefix: prefix string

    Returns:
        Nothing (in-place)
    """
    keys = list(logs_dict.keys())
    for k in keys:
        if k.startswith(prefix):
            cut_key = k.replace(f'{prefix}_', '')
            new_key = f'{prefix}/{cut_key}'
            logs_dict[new_key] = logs_dict.pop(k)
        else:
            logs_dict[f'{prefix}/{k}'] = logs_dict.pop(k)


def rewrite_logs(logs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rewrite logs for compatibility with wandb.

    The logs have their prefix inferred. Then, unnecessary keys are removed, and other keys
    are renamed according to found prefix.

    Args:
        logs_dict: dictionary of logs

    Returns:
        Dictionary of corrected metrics.

    """
    logs_dict = deepcopy(logs_dict)
    is_metric = _is_metric(logs_dict)
    prefix = get_key_prefix(logs_dict) or 'train'

    if is_metric:
        prepare_metrics(logs_dict, prefix)

    rename_keys(logs_dict, prefix)
    return logs_dict


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
        metric_names = ['f1', 'precision', 'recall']
        table_rows = defaultdict(list)
        for metric in metric_names:
            metric_subdict = metrics_dict[f'final_eval_{metric}']
            for dp_name, metric_value in metric_subdict.items():
                if metric_value == "NAN":
                    metric_value = None
                self._wandb.log({f'{dp_name}: {metric}': metric_value})
                table_rows[dp_name].append(metric_value)
        table = [[k] + v for k, v in table_rows.items()]
        columns = ['datapoint'] + metric_names
        self._wandb.log({'metrics': self._wandb.Table(data=table, columns=columns)})

    def _log_predictions(self, predictions):
        if self._wandb is None:
            return
        columns = [
            'Document Name',
            'Entity Name',
            'Predicted Value',
            'Gold Value',
            'Confidence',
            'Is Correct',
        ]
        prediction_data: List[List[Any]] = []
        for doc_path, doc_predictions in predictions.items():
            doc_name = Path(doc_path).name
            for entity_name, entity_values in doc_predictions['entities'].items():
                pred_value = entity_values['text']
                gold_value = entity_values['gold_text']
                confidence = entity_values['avg_confidence']
                is_match = entity_values['is_match']
                row = [
                    doc_name,
                    entity_name,
                    pred_value,
                    gold_value,
                    confidence,
                    is_match,
                ]
                prediction_data.append(row)
        self._wandb.log({'predictions': self._wandb.Table(data=prediction_data, columns=columns)})

    def setup(self, args, state, model, **kwargs):
        super().setup(args, state, model, **kwargs)

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        if self._wandb is None:
            return
        metrics = kwargs.get('metrics').copy()  # type: ignore
        if 'final_eval_predictions' in metrics:
            predictions = metrics.pop('final_eval_predictions')
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
