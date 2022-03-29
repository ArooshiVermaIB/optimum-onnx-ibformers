import logging
from collections import defaultdict
from typing import List, Iterator, Tuple, Union, Sequence, Optional, Dict, Any

import torch
from datasets import Dataset
from typing_extensions import TypedDict
from ibformers.data.chunk import SPLIT_IDX

import numpy as np


class DechunkedDoc(TypedDict):
    """
    Representation of the document reconstructed from the chunking procedure.
    """

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
    """
    Components related to document predictions.
    """

    predicted_classes: np.ndarray  # (seq_length, ), int
    prediction_confidences: np.ndarray  # (seq_length, ), float


class PredictedDoc(DechunkedDoc, PredictionComponents):
    """
    Full representation of the document after predictions.
    """

    pass


def get_predictions_for_sl(predictions: Tuple, dataset: Dataset, label_list: Optional[List] = None) -> Dict[str, Any]:
    """
    Extract prediction dictionary from raw predictions.

    Args:
        predictions: raw predictions of the model with their corresponding labels
        dataset: Chunked dataset that the predictions are from
        label_list: Optional list of label names. If missing, it is inferred from the `labels` col in the dataset.

    Returns:
        Dictionary of document predictions with items:
            * is_test_file: true if the document is test file
            * entities: Dictionary of document-level entities.
    """
    features = dataset.features
    assert "id" in features, "dataset need to contain ids of documents"
    if label_list is None:
        label_list = features["labels"].feature.names
    ids = dataset["id"]
    pred_dict = {}

    for doc_id, chunk_from_idx, chunk_to_idx in doc_chunk_iter(ids):
        # this relies on the fact that each doc chunk contains all non-chunked features,
        # e.g. words or original bounding boxes. That's why we can create prediction for all chunks using only
        # the first doc in the chunk
        doc = prepare_predicted_doc(doc_id, chunk_from_idx, chunk_to_idx, predictions, dataset)

        predicted_entity_words = extract_entity_words(doc["predicted_classes"], doc, label_list, False)
        gold_entity_words = extract_entity_words(doc["gold_labels"], doc, label_list, True)

        doc_dict = create_entities(predicted_entity_words, gold_entity_words, label_list)

        is_test_file = doc["is_test_file"] if "is_test_file" in doc else False
        pred_dict[doc["id"]] = {"is_test_file": is_test_file, "entities": doc_dict}

    return pred_dict


def get_predictions_for_cls(predictions: Tuple, dataset: Dataset, label_list: Optional[List] = None) -> Dict[str, Any]:
    """
    Extract prediction dictionary from raw predictions.

    Args:
        predictions: raw predictions of the model with their corresponding labels
        dataset: Chunked dataset that the predictions are from
        label_list: Optional list of label names. If missing, it is inferred from the `labels` col in the dataset.

    Returns:`
        Dictionary of document predictions with items:
            * is_test_file: true if the document is test file
            * entities: Dictionary of document-level entities.
    """
    features = dataset.features
    id2label = features["labels"]._int2str
    assert "id" in features, "dataset need to contain ids of documents"
    if label_list is None:
        if isinstance(features["labels"], Sequence):
            label_list = features["labels"].feature.names
        else:
            label_list = features["labels"].names
    ids = dataset["id"]
    pred_dict = {}
    pred_logits, _ = predictions

    for doc_id, chunk_from_idx, chunk_to_idx in doc_chunk_iter(ids):
        # this relies on the fact that each doc chunk contains all non-chunked features,
        # e.g. words or original bounding boxes. That's why we can create prediction for all chunks using only
        # the first doc in the chunk
        # doc = prepare_predicted_doc(doc_id, chunk_from_idx, chunk_to_idx, predictions, dataset)
        max_logits_by_lab = defaultdict(list)
        for logits in pred_logits[chunk_from_idx:chunk_to_idx]:
            max_logits_by_lab[logits.argmax()].append(max(logits))
        most_predicted = sorted(
            [(k, len(v), sum(v) / len(v)) for k, v in max_logits_by_lab.items()],
            reverse=True,
            key=lambda x: (x[1], x[2]),
        )

        # Use token page numbers to get the page extremes for cls
        page_nums = dataset[chunk_from_idx:chunk_to_idx]["token_page_nums"]
        page_start = min(page_nums[0])
        page_end = max(page_nums[-1])
        is_test_file = dataset[chunk_from_idx].get("is_test_file", False)  # for non-docpro training
        pred_dict[doc_id] = {
            "is_test_file": is_test_file,
            "entities": {},
            "class_label": id2label[most_predicted[0][0]],
            "class_confidence": most_predicted[0][2],
            "page_start": page_start,
            "page_end": page_end,
        }
    return pred_dict


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


def prepare_predicted_doc(
    doc_id: str, chunk_from_idx: int, chunk_to_idx: int, predictions: Tuple, dataset: Dataset
) -> PredictedDoc:
    """
    Prepare dechunked document with predictions.

    Args:
        doc_id: The doc id that the predictions will be generated from
        chunk_from_idx: Start chunk index of the document
        chunk_to_idx: End chunk index of the document
        predictions: raw predictions of the model with their corresponding labels
        dataset: Chunked dataset that the predictions are from

    Returns:
        Representation of the document with de-chunked data and predictions added.
    """
    doc_components = extract_dechunked_components(doc_id, chunk_from_idx, chunk_to_idx, predictions, dataset)
    prediction_components = calculate_predictions(doc_components["raw_predictions"])
    doc_components.update(**prediction_components)
    return PredictedDoc(**doc_components)


def try_get_first(items: List[Any]):
    if not isinstance(items, list):
        return None
    return items[0]


def extract_dechunked_components(
    doc_id: str, chunk_from_idx: int, chunk_to_idx: int, predictions: Tuple, dataset: Dataset
) -> DechunkedDoc:
    """
    Prepare dechunked document.

    Extracts proper chunks from the dataset and joins the raw predictions.

    Args:
        doc_id: The doc id that the predictions will be generated from
        chunk_from_idx: Start chunk index of the document
        chunk_to_idx: End chunk index of the document
        predictions: raw predictions of the model with their corresponding labels
        dataset: Chunked dataset that the predictions are from

    Returns:
        Representation of the document with de-chunked data and raw predictions added.
    """
    preds, labels = predictions

    doc_spans = dataset[chunk_from_idx:chunk_to_idx]
    assert doc_id == doc_spans["id"][0], "Chunk doc_id and doc_id obtained from the dataset does not match"

    chunk_ranges_lst = doc_spans["chunk_ranges"]
    content_mask_lst = doc_spans["content_tokens_mask"]
    preds_arr = preds[chunk_from_idx:chunk_to_idx]
    labels_arr = labels[chunk_from_idx:chunk_to_idx]
    word_starts = doc_spans["word_starts"]

    doc_preds = join_chunks(preds_arr, chunk_ranges_lst, content_mask_lst)
    doc_labels = join_chunks(labels_arr, chunk_ranges_lst, content_mask_lst)
    doc_word_starts = join_chunks(word_starts, chunk_ranges_lst, None)
    doc_word_starts = np.array(doc_word_starts)

    dechunked_doc = DechunkedDoc(
        id=doc_spans["id"][0],
        words=doc_spans["words"][0],
        word_original_bboxes=try_get_first(doc_spans.get("word_original_bboxes", None)),
        word_page_nums=try_get_first(doc_spans.get("word_page_nums", None)),
        gold_labels=doc_labels[doc_word_starts],
        raw_predictions=doc_preds[doc_word_starts],
        word_starts=doc_word_starts,
        is_test_file=try_get_first(doc_spans.get("is_test_file", False)),
        word_line_idx=try_get_first(doc_spans.get("word_line_idx", None)),
        word_in_line_idx=try_get_first(doc_spans.get("word_in_line_idx", None)),
    )
    return dechunked_doc


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

    # create temporary matrix to do avaraging of overlapping parts -
    # we make assumption that for given token could belong only to 2 chunks
    doc_arr = np.full((2, max_size, *chunk_shape[1:]), fill_value=np.nan)
    for i, (chunk, rng, content_mask) in enumerate(zip(chunks, chunk_ranges, content_mask_lst)):
        is_even = i % 2
        rng_len = rng[1] - rng[0]
        # for the last chunk there might be padding so content mask will have different length
        if content_mask is None:
            content_chunk = chunk
        else:
            content_mask_with_padding = content_mask + [False] * (len(chunk) - len(content_mask))
            content_chunk = chunk[content_mask_with_padding]

        assert len(content_chunk) == rng_len, "Length of content in the chunk should be equal to chunk range length"
        doc_arr[is_even, rng[0] : rng[1]] = content_chunk

    doc_arr_mean = np.nanmean(doc_arr, axis=0)

    return doc_arr_mean.astype(first_chunk.dtype)


def calculate_predictions(raw_predictions: np.ndarray) -> PredictionComponents:
    """
    Calculate probabilities and selected classes from raw predictions.


    Args:
        raw_predictions: (num_samples x num_classes) numpy array

    Returns:
        Dictionary with predicted classes and their confidences.
    """
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


def create_entities(
    predicted_entity_words: Dict[str, List[Dict[str, Any]]],
    gold_entity_words: Dict[str, List[Dict[str, Any]]],
    label_list: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Create entities from predicted and gold entity words.

    Currently we define "entity" as all text labeled as given entity type. It can as well be
    a whole column of entities, or multiple occurrences of the same value.

    This function requires at least the `raw_word` and `conf` (for predictions) keys present in entity word dicts.
    It does however puts all the word dict contents inside the entities.

    Args:
        predicted_entity_words: Predicted entities grouped by label name
        gold_entity_words: Gold entities grouped by label name
        label_list: list of labels

    Returns:
        Dictionary of entity data grouped by label name.
    """
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
    return doc_dict


def extract_entity_words(
    prediction_idxs: np.ndarray, doc: Dict[str, Any], label_list: List[str], is_gold: bool
) -> Dict[str, Any]:
    """
    Given predicted classes, create a dictionary with entity words.

    Args:
        prediction_idxs:
        doc:
        label_list:
        is_gold:

    Returns:

    """
    non_zero_class = np.nonzero(prediction_idxs)[0]
    doc_words_dict = defaultdict(list)
    for idx in non_zero_class:
        class_idx = prediction_idxs[idx]
        conf = 0 if is_gold else doc["prediction_confidences"][idx]
        tag_name = label_list[class_idx]
        if "word_original_bboxes" in doc and doc["word_original_bboxes"] is not None:
            org_bbox = doc["word_original_bboxes"][idx]
        else:
            org_bbox = [0, 0, 0, 0]
        if "word_page_nums" in doc and doc["word_page_nums"] is not None:
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
        if not is_gold:
            if doc.get("word_line_idx", None) is not None:
                word["word_line_idx"] = doc["word_line_idx"][idx]
            if doc.get("word_in_line_idx", None) is not None:
                word["word_in_line_idx"] = doc["word_in_line_idx"][idx]

        doc_words_dict[tag_name].append(word)
    return doc_words_dict


def get_predictions_for_sc(predictions: Tuple, dataset: Dataset):
    ids = dataset["id"]
    pages = dataset["page"]
    is_test_file = dataset["is_test_file"]
    class_id2str = dataset.features["class_label"].feature._int2str
    split_idx = SPLIT_IDX

    test_doc_dict = {}

    doc_dict = defaultdict(list)
    for i, (id, page, test_file) in enumerate(zip(ids, pages, is_test_file)):
        doc_dict[id].append((i, page))
        test_doc_dict[id] = test_file

    final_predictions = {}
    for id in doc_dict:
        doc_dict[id].sort(key=lambda x: x[1])
        page_nums = [x[1] for x in doc_dict[id]]
        page_offsets = [x[0] for x in doc_dict[id]]
        preds = predictions.predictions[page_offsets, :]
        splits = np.argmax(preds[:, 0:2], axis=1)

        start_range = 0
        end_range = 0
        predicted_dict = defaultdict(list)
        for i, split in enumerate(splits):
            if split == split_idx or i == len(splits) - 1:
                end_range = i
                cur_preds = preds[start_range : end_range + 1, :]
                split_conf = torch.tensor(cur_preds[-1, :2]).float().softmax(0)[split_idx]
                avg_class_pred = torch.tensor(cur_preds[:, 2:]).float().softmax(1).sum(0).softmax(0)
                class_idx = avg_class_pred.argmax().item()
                class_conf = avg_class_pred[class_idx].item()
                class_name = class_id2str[class_idx]
                pred_tuple = [page_nums[start_range] + 1, page_nums[end_range] + 1, class_conf, float(split_conf)]

                predicted_dict[class_name].append(pred_tuple)
                start_range = i + 1
        final_predictions[id] = {"is_test_file": test_doc_dict[id], "prediction": dict(predicted_dict)}

    return final_predictions
