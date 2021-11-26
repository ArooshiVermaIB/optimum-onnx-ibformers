import logging
from collections import defaultdict
from typing import Tuple, Iterator, Union, List, Sequence, Mapping, Dict, Optional, Any

import numpy as np
import pandas as pd
import torch
from datasets import load_metric, Dataset
from typing_extensions import TypedDict


def doc_chunk_iter(doc_ids: List[str]) -> Iterator[Tuple[str, int, int]]:
    """
    function will return list of doc_ids with ranges of chunks corresponding to given doc_id
    :param doc_ids: list of document ids
    :return: Iterator of tuples
    """
    from_idx = 0
    next_doc_ids = doc_ids[1:] + ["*end*"]
    for i, (doc_id, next_doc_id) in enumerate(zip(doc_ids, next_doc_ids)):
        if doc_id != next_doc_id:
            yield doc_id, from_idx, i + 1
            from_idx = i + 1


def join_chunks(
    chunks: Union[List[Sequence], np.ndarray],
    chunk_ranges: List[Sequence[int]],
    content_mask_lst: Optional[List[Sequence[int]]] = None,
) -> np.ndarray:
    """
    When we get predictions for overlapping chunks of an input sequence, we have to combine the predictions, doing
    something with the overlap. In this function, we simply take the feature-wise mean for each of the tokens that
    have predictions from multiple chunks.
    Join list of ndarray of chunks based on the information in the chunk_ranges list
    :param chunks: Chunks which need to be joined
    :param chunk_ranges: Sequence of ranges which inform about position of chunk in the original document
    :param content_mask_lst: List of content_mask for each chunk. content_mask is a Sequence of booleans.
        If not passed, content mask won't be used
    :return:
    """

    if content_mask_lst is None:
        content_mask_lst = [None] * len(chunks)
    strictly_increasing = all(i[0] < j[0] for i, j in zip(chunk_ranges, chunk_ranges[1:]))
    assert strictly_increasing, f"Ranges of the chunks seems incorrect: Value: {chunk_ranges}"
    max_size = chunk_ranges[-1][-1]
    first_chunk = np.array(chunks[0])
    chunk_shape = first_chunk.shape
    doc_arr = np.full((len(chunks), max_size, *chunk_shape[1:]), fill_value=np.nan)
    for i, (chunk, rng, content_mask) in enumerate(zip(chunks, chunk_ranges, content_mask_lst)):
        rng_len = rng[1] - rng[0]
        # for the last chunk there might be padding so content mask will have different length
        if content_mask is None:
            content_chunk = chunk
        else:
            content_mask_with_padding = content_mask + [False] * (len(chunk) - len(content_mask))
            content_chunk = chunk[content_mask_with_padding]

        assert len(content_chunk) == rng_len, "Length of content in the chunk should be equal to chunk range length"
        doc_arr[i, rng[0] : rng[1]] = content_chunk

    doc_arr_mean = np.nanmean(doc_arr, axis=0)

    return doc_arr_mean.astype(first_chunk.dtype)


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


class DechunkedDoc(TypedDict):
    id: str
    words: Sequence[str]
    word_original_bboxes: Optional[Sequence[Tuple[int, int, int, int]]]
    word_page_nums: Optional[Sequence[int]]
    gold_labels: np.ndarray  # (seq_length, ), int
    raw_predictions: np.ndarray  # (seq_length, num_labels), float
    word_starts: np.ndarray  # (seq_length, ), bool
    is_test_file: bool
    word_line_idx: Optional[np.ndarray]
    word_in_line_idx: Optional[np.ndarray]


class PredictionComponents(TypedDict):
    predicted_classes: np.ndarray  # (seq_length, ), int
    prediction_confidences: np.ndarray  # (seq_length, ), float


class PredictedDoc(DechunkedDoc, PredictionComponents):
    pass


def get_predictions_for_sl(predictions: Tuple, dataset: Dataset, label_list: Optional[List] = None):
    features = dataset.features
    assert "id" in features, "dataset need to contain ids of documents"
    if label_list is None:
        label_list = features["labels"].feature.names
    preds, labels = predictions
    ids = dataset["id"]
    pred_dict = {}

    for doc_id, chunk_from_idx, chunk_to_idx in doc_chunk_iter(ids):
        # this relies on the fact that each doc chunk contains all non-chunked features,
        # e.g. words or original bounding boxes. That's why we can create prediction for all chunks using only
        # the first doc in the chunk
        doc: PredictedDoc = prepare_predicted_doc(dataset, doc_id, chunk_from_idx, chunk_to_idx, predictions)

        predicted_entity_words = extract_entity_words(doc["predicted_classes"], doc, label_list, True)
        gold_entity_words = extract_entity_words(doc["gold_labels"], doc, label_list, False)

        doc_dict = {}
        for k in label_list[1:]:
            pred_words = predicted_entity_words.get(k, [])
            pred_text = " ".join([w["raw_word"] for w in pred_words])
            gold_words = gold_entity_words.get(k, [])
            gold_text = " ".join([w["raw_word"] for w in gold_words])
            doc_dict[k] = {
                "words": pred_words,
                "text": pred_text,
                "avg_confidence": np.mean([w["conf"] for w in pred_words]),
                "gold_text": gold_text,
                "gold_words": gold_words,
                "is_match": pred_text == gold_text,
            }

        is_test_file = doc["is_test_file"] if "is_test_file" in doc else False
        pred_dict[doc["id"]] = {"is_test_file": is_test_file, "entities": doc_dict}

    return pred_dict


def extract_entity_words(prediction_idxs, doc, label_list, is_gold):
    non_zero_class = np.nonzero(prediction_idxs)[0]
    doc_words_dict = defaultdict(list)
    for idx in non_zero_class:
        class_idx = prediction_idxs[idx]
        conf = 0 if is_gold else doc["prediction_confidences"][idx]
        tag_name = label_list[class_idx]
        if "word_original_bboxes" in doc:
            org_bbox = doc["word_original_bboxes"][idx]
        else:
            org_bbox = [0, 0, 0, 0]
        if "word_page_nums" in doc:
            page = doc["word_page_nums"][idx]
        else:
            page = 0
        word = dict(
            raw_word=doc["words"][idx],
            start_x=org_bbox[0],
            start_y=org_bbox[1],
            end_x=org_bbox[2],
            end_y=org_bbox[3],
            line_height=org_bbox[3] - org_bbox[1],
            word_width=org_bbox[2] - org_bbox[0],
            page=page,
            conf=conf,
            idx=idx,
        )
        if is_gold:
            if doc["word_line_idx"] is not None:
                word["word_line_idx"] = doc["word_line_idx"][idx]
            if doc["word_in_line_idx"] is not None:
                word["word_in_line_idx"] = doc["word_in_line_idx"][idx]

        doc_words_dict[tag_name].append(word)
    return doc_words_dict


def extract_dechunked_components(dataset, doc_id, chunk_from_idx, chunk_to_idx, predictions) -> DechunkedDoc:
    preds, labels = predictions

    doc = dataset[chunk_from_idx]
    assert doc_id == doc["id"], "Chunk doc_id and doc_id obtained from the dataset does not match"

    chunk_ranges = dataset["chunk_ranges"]
    chunk_ranges_lst = chunk_ranges[chunk_from_idx:chunk_to_idx]
    content_mask_lst = dataset["content_tokens_mask"][chunk_from_idx:chunk_to_idx]
    preds_arr = preds[chunk_from_idx:chunk_to_idx]
    labels_arr = labels[chunk_from_idx:chunk_to_idx]
    word_starts = dataset["word_starts"][chunk_from_idx:chunk_to_idx]

    doc_preds = join_chunks(preds_arr, chunk_ranges_lst, content_mask_lst)
    doc_labels = join_chunks(labels_arr, chunk_ranges_lst, content_mask_lst)
    doc_word_starts = join_chunks(word_starts, chunk_ranges_lst, None)
    doc_word_starts = np.array(doc_word_starts)

    dechunked_doc = DechunkedDoc(
        id=doc["id"],
        words=doc["words"],
        word_original_bboxes=doc.get("word_original_bboxes", None),
        word_page_nums=doc.get("word_page_nums", None),
        gold_labels=doc_labels[doc_word_starts],
        raw_predictions=doc_preds[doc_word_starts],
        word_starts=doc_word_starts,
        is_test_file=doc.get("is_test_file", False),
        word_line_idx=doc.get("word_line_idx", None),
        word_in_line_idx=doc.get("word_in_line_idx", None),
    )
    return dechunked_doc


def calculate_predictions(raw_predictions: np.ndarray, word_starts: np.ndarray) -> PredictionComponents:
    # softmax for np
    doc_prob = torch.tensor(raw_predictions.astype(np.float)).softmax(1).numpy()
    doc_conf = np.max(doc_prob, axis=-1)
    if np.isnan(doc_conf).sum() > 0:
        logging.warning(
            "There are NaNs in the model predictions. "
            "If this run is using mixed precision try to run it with single precision"
        )

    doc_class_index = np.argmax(doc_prob, axis=-1)
    return PredictionComponents(predicted_classes=doc_class_index, prediction_confidences=doc_conf)


def prepare_predicted_doc(dataset, doc_id, chunk_from_idx, chunk_to_idx, predictions) -> PredictedDoc:
    doc_components = extract_dechunked_components(dataset, doc_id, chunk_from_idx, chunk_to_idx, predictions)
    prediction_components = calculate_predictions(doc_components["raw_predictions"], doc_components["word_starts"])
    doc_components.update(**prediction_components)
    return PredictedDoc(**doc_components)


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
