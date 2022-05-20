import logging
from typing import Tuple, List, Mapping, Dict, Optional, Union, Any

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import classification_report, accuracy_score

from ibformers.data.chunk import NO_SPLIT_IDX, SPLIT_IDX
from ibformers.data.predict import get_predictions_for_sl, get_predictions_for_cls, get_predictions_for_sc
from ibformers.data.predict_qa import get_predictions_for_qa

logger = logging.getLogger(__name__)


def iou_score(
    y_true: Mapping[str, List[int]], y_pred: Mapping[str, List[int]], all_tags: List[str]
) -> Dict[str, float]:
    result = {}
    for t in all_tags:
        if t == "O":
            continue
        if (t not in y_pred) and (t not in y_true):
            result[t] = 1.0
            continue
        elif (t not in y_pred) or (t not in y_true):
            result[t] = 0.0
            continue
        a = set(y_pred[t])
        b = set(y_true[t])
        union = len(a.union(b))
        intersection = len(a.intersection(b))
        result[t] = (intersection / union) if union > 0 else 1.0
    return result


def calculate_average_metrics(token_level_df: pd.DataFrame) -> Dict[str, Union[float, str]]:
    """
    Calculate average (micro- and macro-) metrics.

    Args:
        token_level_df: Dataframe with columns `true_positives`,
            `total_positives`, `total_true`, `f1`, `precision`, `recall`

    Returns:
        Dictionary with micro- and macro- f1, precision and recall.
    """

    summed_df = token_level_df.sum(axis=0)
    summed_df["micro_precision"] = summed_df["true_positives"] / summed_df["total_positives"]
    summed_df["micro_recall"] = summed_df["true_positives"] / summed_df["total_true"]
    summed_df.fillna(0, inplace=True)
    summed_df["micro_f1"] = (2 * summed_df["micro_precision"] * summed_df["micro_recall"]) / (
        summed_df["micro_precision"] + summed_df["micro_recall"] + 1e-10
    )

    average_results = summed_df[["micro_precision", "micro_recall", "micro_f1"]].to_dict()

    # ignore fields with no gold values
    macro_metrics_df = token_level_df[token_level_df["total_true"] > 0].fillna(0)
    if macro_metrics_df.shape[0] == 0:
        average_results["macro_f1"] = average_results["macro_precision"] = average_results["macro_recall"] = "NAN"
    else:
        average_results["macro_f1"] = macro_metrics_df["f1"].mean()
        average_results["macro_precision"] = macro_metrics_df["precision"].mean()
        average_results["macro_recall"] = macro_metrics_df["recall"].mean()
    return average_results


def compute_legacy_metrics(label_list: List[str], pred_dict: Dict[str, Any]) -> Dict[str, Any]:
    ground_truths: List[Dict[str, List]] = [
        {k: [wrd["idx"] for wrd in v["gold_words"]] for k, v in doc_lab["entities"].items()}
        for doc_lab in pred_dict.values()
    ]
    pred_words: List[Dict[str, List]] = [
        {k: [wrd["idx"] for wrd in v["words"]] for k, v in doc_lab["entities"].items()}
        for doc_lab in pred_dict.values()
    ]

    token_level: Mapping[str, Mapping[str, int]] = {
        k: {"true_positives": 0, "total_positives": 0, "total_true": 0} for k in label_list if k != "O"
    }

    doc_level_iou: List[Mapping[str, float]] = []
    for y_true, y_pred in zip(ground_truths, pred_words):
        # Throw away the confidence number
        for t in label_list:
            if t == "O":
                continue
            a = set(y_pred.get(t, []))
            b = set(y_true.get(t, []))
            token_level[t]["total_positives"] += len(a)
            token_level[t]["total_true"] += len(b)
            token_level[t]["true_positives"] += len(a.intersection(b))
        iou = iou_score(y_true, y_pred, label_list)
        doc_level_iou.append(iou)

    iou_df = pd.DataFrame(doc_level_iou)
    # TODO: Add other metrics and make customizable
    doc_level_metrics: Mapping[str, Mapping[str, float]] = {
        "exact_match": (iou_df == 1).mean().to_dict(),  # type: ignore
    }

    token_level_df = pd.DataFrame(token_level).T
    token_level_df["precision"] = token_level_df.true_positives / token_level_df.total_positives
    token_level_df["recall"] = token_level_df.true_positives / token_level_df.total_true
    token_level_df["f1"] = (
        2
        * token_level_df.precision
        * token_level_df.recall
        / (token_level_df.precision + token_level_df.recall + 1e-10)
        # Note that this is Pandas, so dividing by zero gives NAN
    )

    average_results = calculate_average_metrics(token_level_df)

    logger.info("EVALUATION RESULTS")
    logger.info(token_level_df)
    token_level_results = token_level_df.fillna("NAN")[["precision", "recall", "f1"]].to_dict()
    results = {**doc_level_metrics, **token_level_results, **average_results}

    results["predictions"] = pred_dict
    return results


def compute_legacy_metrics_for_sl(
    predictions: Tuple, dataset: Dataset, label_list: Optional[List] = None
) -> Dict[str, Any]:
    """
    Compute metrics for sequence labelling.

    The computed metrics are:
    * exact_match (per field name)
    * precision, recall, f1 (per field name)
    * precision, recall, f1 (micro- and macro- averaged)

    Args:
        predictions: Tuple of prediciton, labels tensors
        dataset: the dataset where the predictions come from
        label_list: Optional list of label names. If none provided, it will be inferred from `labels` column in the
         dataset

    Returns:

    """

    if label_list is None:
        label_list = dataset.features["labels"].feature.names
    # get prediction dict and print mismatches
    pred_dict = get_predictions_for_sl(predictions, dataset, label_list)

    max_examples = 2
    max_len = 50
    mismatches_text = f"MISMATCH EXAMPLES (max {max_examples} per label)\n"
    for lab in label_list[1:]:
        mismatches = []
        for k, v in pred_dict.items():
            if len(mismatches) == max_examples:
                break
            if not v["entities"][lab]["is_match"]:
                pred_text = get_log_entity_text(v["entities"][lab]["text"], max_len)
                lab_text = get_log_entity_text(v["entities"][lab]["gold_text"], max_len)
                mismatches.append("\tpred:\t'" + pred_text + "'\n\tgold:\t'" + lab_text + "'\n")

        if len(mismatches) > 0:
            label_mismatch_text = "  ".join(mismatches)
            mismatches_text += f"{lab}:\n{label_mismatch_text}\n"
    logger.info(mismatches_text)

    results = compute_legacy_metrics(label_list, pred_dict)
    return results


def get_log_entity_text(text: str, max_len: int) -> str:
    # shorten the text to be better presented on logs
    return text if len(text) < max_len else text[: max_len // 2] + " [...] " + text[: max_len // 2]


def compute_only_legacy_metrics_for_sl(
    predictions: Tuple, dataset: Dataset, label_list: Optional[List] = None
) -> Dict[str, Any]:
    results = compute_legacy_metrics_for_sl(predictions, dataset, label_list)
    results.pop("predictions")
    return results


def compute_legacy_metrics_for_mqa(predictions: Tuple, dataset: Dataset):
    """
    Function will recompute predictions and labels from extra token head to match sequence labeling format
    :param predictions:
    :param dataset:
    :return:
    """

    preds, labels = predictions
    new_preds, new_labels = [], []
    last_used_label_ids = None
    for pred, lab, doc in zip(preds, labels, dataset):

        # for performance reasons, only the first chunk in document contains "global"
        # columns, such as "entities"
        if doc["entities"]["used_label_id"] is None:
            label_ids = last_used_label_ids
        else:
            label_ids = doc["entities"]["used_label_id"]
            last_used_label_ids = doc["entities"]["used_label_id"]
        reorder_index = np.array([0] + label_ids)
        new_pred = pred[:, reorder_index]
        map_dict = {v: idx for idx, v in enumerate(reorder_index)}
        map_dict[-100] = -100
        new_lab = np.array([map_dict[l] for l in lab])

        new_preds.append(new_pred)
        new_labels.append(new_lab)

    new_predictions = (np.stack(new_preds), np.stack(new_labels))

    return compute_legacy_metrics_for_sl(new_predictions, dataset)


def compute_metrics_for_qa_task(predictions: Tuple, dataset: Dataset):
    """
    Function will create dummy label list to compute metrics for token classification task
    :param predictions:
    :param dataset:
    :return:
    """
    num_labels = predictions[0].shape[-1]
    dummy_label_list = [f"class_{i}" for i in range(num_labels)]

    return compute_legacy_metrics_for_sl(predictions, dataset, dummy_label_list)


def compute_metrics_for_singleqa_task(predictions: Tuple, dataset: Dataset):
    """
    Function will create dummy label list to compute metrics for token classification task
    :param predictions:
    :param dataset:
    :return:
    """
    label_list = dataset.features["labels"].feature.names
    pred_dict = get_predictions_for_qa(predictions, dataset)
    results = compute_legacy_metrics(label_list, pred_dict)
    return results


def compute_metrics_for_sl_grp(predictions: Tuple, dataset: Dataset):
    """
    Function will create dummy label list to compute metrics for token classification task
    :param predictions:
    :param dataset:
    :return:
    """

    preds, labels = predictions
    num_classes = int(preds.shape[-1] / 3)
    row_preds, col_preds, table_preds = (
        preds[:, :, :num_classes],
        preds[:, :, num_classes:-num_classes],
        preds[:, :, -num_classes:],
    )
    row_labels, col_labels, table_labels = labels[:, :, 0], labels[:, :, 1], labels[:, :, 2]

    row_predictions = row_preds, row_labels
    col_predictions = col_preds, col_labels
    table_predictions = table_preds, table_labels
    row_label_list = ["O"] + [f"row_{i}" for i in range(num_classes)]
    col_label_list = ["O"] + [f"col_{i}" for i in range(num_classes)]
    table_label_list = ["O"] + [f"table_{i}" for i in range(num_classes)]
    row_metrics = compute_legacy_metrics_for_sl(row_predictions, dataset, row_label_list)
    column_metrics = compute_legacy_metrics_for_sl(col_predictions, dataset, col_label_list)
    table_metrics = compute_legacy_metrics_for_sl(table_predictions, dataset, table_label_list)

    for k, v in row_metrics.items():
        column_metrics[f"row_{k}"] = v
    for k, v in table_metrics.items():
        column_metrics[f"table_{k}"] = v
    return column_metrics


def _add_training_loop_required_metrics(metrics_dict: Dict[str, Any]) -> None:
    metrics = metrics_dict["metrics"]
    metrics_dict.update(
        {
            "macro_f1": metrics["f1"]["macro avg"],
            "macro_recall": metrics["precision"]["macro avg"],
            "macro_precision": metrics["recall"]["macro avg"],
        }
    )

    # when micro avg are not equal to accuracy, sklearn adds it to the report
    if "micro avg" in metrics["f1"]:
        metrics_dict.update(
            {
                "micro_f1": metrics["f1"]["micro avg"],
                "micro_recall": metrics["precision"]["micro avg"],
                "micro_precision": metrics["recall"]["micro avg"],
            }
        )
    else:
        accuracy = metrics["accuracy"]
        metrics_dict.update(
            {
                "micro_f1": accuracy,
                "micro_recall": accuracy,
                "micro_precision": accuracy,
            }
        )


def compute_metrics_for_cls(predictions: Tuple, dataset: Dataset):
    preds, labels = predictions
    pred_lab = preds.argmax(-1)

    id2label = dataset.features["labels"]._int2str

    metrics = _compute_classification_metrics(labels, pred_lab, id2label)

    return_dict = {}
    return_dict["metrics"] = metrics

    return_dict["predictions"] = get_predictions_for_cls(predictions, dataset)

    _add_training_loop_required_metrics(return_dict)

    return return_dict


def compute_metrics_for_sc(predictions: Tuple, dataset: Dataset):

    class_id2str = dataset.features["class_label"].feature._int2str
    split_id2str = {SPLIT_IDX: "split", NO_SPLIT_IDX: "no-split"}

    splitter_labels = predictions.label_ids[:, 0]
    classifier_labels = predictions.label_ids[:, 1]

    splitter_preds = predictions.predictions[:, 0:2]
    classifier_preds = predictions.predictions[:, 2:]

    splits = np.argmax(splitter_preds, axis=1)
    classes = np.argmax(classifier_preds, axis=1)

    splitter_metrics = _compute_classification_metrics(splitter_labels, splits, split_id2str)

    classifier_metrics = _compute_classification_metrics(classifier_labels, classes, class_id2str)

    metrics = {"splitter_metrics": splitter_metrics, "classifier_metrics": classifier_metrics}
    metrics["predictions"] = get_predictions_for_sc(predictions, dataset)

    return metrics


def _compute_classification_metrics(y_true, y_pred, id2label):

    labels = list(range(len(id2label)))
    target_names = [id2label[label] for label in labels]
    metrics = classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True)
    if "accuracy" in metrics:
        metrics.pop("accuracy")

    metrics = pd.DataFrame(metrics).T.to_dict()

    # to keep it as in extraction
    metrics["f1"] = metrics.pop("f1-score")

    metrics.update({"accuracy": accuracy_score(y_true, y_pred)})

    return metrics
