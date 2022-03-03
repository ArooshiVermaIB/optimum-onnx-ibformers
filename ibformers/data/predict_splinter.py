import logging
from typing import Tuple, Dict, Text

import numpy as np
import torch
from datasets import Dataset, load_metric

from ibformers.data.metrics import compute_legacy_metrics
from ibformers.data.predict import (
    PredictionComponents,
    doc_chunk_iter,
    extract_dechunked_components,
    extract_entity_words,
    create_entities,
)
from ibformers.data.predict_qa import _create_question_chunks, extract_answer_words, get_qa_predictions
from ibformers.models.layv1splinter import SPLINTER_MAX_QUESTIONS


def calculate_bin_predictions(raw_predictions: np.ndarray) -> PredictionComponents:
    """
    Calculate probabilities and selected classes from raw predictions.


    Args:
        raw_predictions: (num_samples) numpy array

    Returns:
        Dictionary with predicted classes and their confidences.
    """
    # sigmoid for np
    doc_conf = torch.tensor(raw_predictions.astype(np.float)).sigmoid().numpy()
    doc_class_index = (doc_conf > 0.5).astype(np.int)
    if np.isnan(doc_conf).sum() > 0:
        logging.warning(
            "There are NaNs in the model predictions. "
            "If this run is using mixed precision try to run it with single precision"
        )
    return PredictionComponents(predicted_classes=doc_class_index, prediction_confidences=doc_conf)


def get_predictions_for_splinter_mqa(predictions: Tuple, dataset: Dataset) -> Dict[Text, Dict]:
    """
    Extract prediction dictionary from raw predictions for splinter sl task
    """
    label_list = dataset.features["labels"].feature.names
    features = dataset.features
    assert "id" in features, "dataset need to contain ids of documents"
    ids = dataset["id"]
    pred_dict = {}

    # Iterate through doc chunks.
    for doc_id, chunk_from_idx, chunk_to_idx in doc_chunk_iter(ids):
        assert (
            doc_id == dataset[chunk_from_idx]["id"]
        ), "Chunk doc_id and doc_id obtained from the dataset does not match"

        predicted_words = {}
        gold_words = {}
        # Iterate through question chunks.
        for qchunk in _create_question_chunks(dataset, chunk_from_idx, chunk_to_idx):
            doc = dataset[qchunk["from_idx"]]
            doc_dechunked = extract_dechunked_components(
                doc_id, qchunk["from_idx"], qchunk["to_idx"], predictions, dataset
            )

            for qidx, question in enumerate(doc["entities"]["name"]):
                prediction_components = calculate_bin_predictions(doc_dechunked["raw_predictions"][:, qidx])
                dummy_label_list = ["0", "1"]
                labels = doc_dechunked["gold_labels"][:, qidx]
                doc_for_question = {**doc_dechunked, **prediction_components, "gold_labels": labels}
                predicted_words[question] = extract_entity_words(
                    prediction_components["predicted_classes"], doc_for_question, dummy_label_list, False
                )["1"]
                gold_words[question] = extract_entity_words(labels, doc_for_question, dummy_label_list, True)["1"]

        doc_dict = create_entities(predicted_words, gold_words, label_list)
        is_test_file = doc["is_test_file"] if "is_test_file" in doc else False
        pred_dict[doc["id"]] = {"is_test_file": is_test_file, "entities": doc_dict}
    return pred_dict


def squad_metric(p: Tuple, dataset):
    """
    Load squad metric and build the input required for this metric
    """
    metric = load_metric("squad")
    ids = dataset["qid"]

    preds = []
    golds = []
    for doc_id, chunk_from_idx, chunk_to_idx in doc_chunk_iter(ids):
        doc = dataset[chunk_from_idx]
        words = doc["words"]
        qchunk = dict(from_idx=chunk_from_idx, to_idx=chunk_to_idx)
        pred_ans, gold_ans = get_qa_predictions(p, dataset, qchunk)
        if pred_ans["end"] > pred_ans["start"]:
            pred_text = " ".join(words[pred_ans["start"] : pred_ans["end"]])
        else:
            pred_text = ""
        gold_text = " ".join(words[gold_ans["start"] : gold_ans["end"]])

        preds.append({"id": doc_id, "prediction_text": pred_text})
        golds.append({"id": doc_id, "answers": [{"text": gold_text, "answer_start": 0}]})

    return metric.compute(predictions=preds, references=golds)


def splinter_metric(p: Tuple, dataset):
    """
    Dummy metric which check average IOU for all the answers for the SL task. Used for pretraining on public datasets
    """
    preds = p.predictions
    labels = p.label_ids

    ious = []
    for pred, lab in zip(preds, labels):

        for q in range(SPLINTER_MAX_QUESTIONS):
            qpred = pred[:, q]
            qlab = lab[:, q]

            num_pos = (qlab > 0).sum()

            if num_pos == 0:
                break

            notignore_mask = qlab != -100
            qpred_binary = (qpred > 0).astype(np.int)

            qpred_fl = qpred_binary[notignore_mask]
            qlab_fl = qlab[notignore_mask]

            qpred_nz = qpred_fl.nonzero()[0]
            qlab_nz = qlab_fl.nonzero()[0]

            pred_set = set(qpred_nz)
            lab_set = set(qlab_nz)

            _union = len(pred_set.union(lab_set))
            _intersection = len(pred_set.intersection(lab_set))

            iou = (_intersection / _union) if _union > 0 else np.nan

            ious.append(iou)

    iou_score = np.array(ious).mean()
    print(f"IOU: {iou_score}")

    return {"iou": iou_score}


def compute_metrics_for_splinter_mqa(predictions: Tuple, dataset: Dataset):
    """
    Function will compute predictions based on the output from splinter which is binary vector indicating
    answer positions for eqch question separately
    """

    label_list = dataset.features["labels"].feature.names
    pred_dict = get_predictions_for_splinter_mqa(predictions, dataset)
    results = compute_legacy_metrics(label_list, pred_dict)
    return results
