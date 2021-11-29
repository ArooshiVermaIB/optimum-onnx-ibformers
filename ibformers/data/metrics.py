import logging
from typing import Tuple, List, Mapping, Dict, Optional

import numpy as np
import pandas as pd
from datasets import Dataset

from ibformers.data.predict import get_predictions_for_sl


def iou_score(y_true: Mapping[str, List[int]], y_pred: Mapping[str, List[int]], all_tags: List[str]) -> Dict[str, int]:
    result = {}
    for t in all_tags:
        if t == "O":
            continue
        if (t not in y_pred) or (t not in y_true):
            result[t] = 0
            continue
        a = set(y_pred[t])
        b = set(y_true[t])
        _union = len(a.union(b))
        _intersection = len(a.intersection(b))
        result[t] = (_intersection / _union) if _union > 0 else 0.0
    return result


def calculate_average_metrics(token_level_df: pd.DataFrame) -> Dict[str, float]:
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
    macro_metrics_df = token_level_df[token_level_df["total_positives"] > 0].fillna(0)
    if macro_metrics_df.shape[0] == 0:
        average_results["macro_f1"] = average_results["macro_precision"] = average_results["macro_recall"] = "NAN"
    else:
        average_results["macro_f1"] = macro_metrics_df["f1"].mean()
        average_results["macro_precision"] = macro_metrics_df["precision"].mean()
        average_results["macro_recall"] = macro_metrics_df["recall"].mean()
    return average_results


def compute_legacy_metrics_for_sl(predictions: Tuple, dataset: Dataset, label_list: Optional[List] = None):

    if label_list is None:
        label_list = dataset.features["labels"].feature.names
    # get prediction dict and print mismatches
    pred_dict = get_predictions_for_sl(predictions, dataset, label_list)

    max_examples = 2
    mismatches_text = f"MISMATCH EXAMPLES (max {max_examples} per label)\n"
    for lab in label_list[1:]:
        mismatches = [
            "\tpred:\t'" + v["entities"][lab]["text"] + "'\n\tgold:\t'" + v["entities"][lab]["gold_text"] + "'\n"
            for k, v in pred_dict.items()
            if not v["entities"][lab]["is_match"]
        ]
        label_mismatch_text = "  ".join(mismatches[:max_examples])
        if len(mismatches) > 0:
            mismatches_text += f"{lab}:\n{label_mismatch_text}\n"
    logging.info(mismatches_text)

    # get list of document gold labels - List[Dict[List]]
    ground_truths: List[Dict[List]] = [
        {k: [wrd["idx"] for wrd in v["gold_words"]] for k, v in doc_lab["entities"].items()}
        for doc_lab in pred_dict.values()
    ]
    pred_words: List[Dict[List]] = [
        {k: [wrd["idx"] for wrd in v["words"]] for k, v in doc_lab["entities"].items()}
        for doc_lab in pred_dict.values()
    ]

    token_level: Mapping[str, Mapping[str, int]] = {
        k: {"true_positives": 0, "total_positives": 0, "total_true": 0} for k in label_list if k != "O"
    }

    doc_level_results: List[Mapping[str, int]] = []
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
        doc_level_results.append(iou)

    df = pd.DataFrame(doc_level_results)

    # TODO: Add other metrics and make customizable
    doc_level_metrics: Mapping[str, Mapping[str, float]] = {
        "exact_match": (df == 1).mean().to_dict(),
    }
    overall_accuracy = (df == 1).mean().mean()

    token_level_df = pd.DataFrame(token_level).T
    token_level_df["precision"] = token_level_df.true_positives / token_level_df.total_positives
    token_level_df["recall"] = token_level_df.true_positives / token_level_df.total_true
    token_level_df["f1"] = (
        2
        * token_level_df.precision
        * token_level_df.recall
        / (token_level_df.precision + token_level_df.recall)
        # Note that this is Pandas, so dividing by zero gives NAN
    )

    average_results = calculate_average_metrics(token_level_df)

    logging.info("EVALUATION RESULTS")
    logging.info(token_level_df)
    token_level_results = token_level_df.fillna("NAN")[["precision", "recall", "f1"]].to_dict()
    results = {**doc_level_metrics, **token_level_results, **average_results}

    results["predictions"] = pred_dict

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

    for pred, lab, doc in zip(preds, labels, dataset):
        reorder_index = np.array([0] + doc["entities"]["used_label_id"])
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
