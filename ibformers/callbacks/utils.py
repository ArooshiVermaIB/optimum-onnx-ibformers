import logging
from copy import deepcopy
from typing import Dict, Any, Optional

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

