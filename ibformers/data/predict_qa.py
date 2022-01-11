import logging
from typing import Any, Dict, List, Text, Tuple

from typing_extensions import TypedDict

from datasets import Dataset
import numpy as np

from ibformers.data.predict import doc_chunk_iter, calculate_predictions, join_chunks, create_entities


class QAPrediction(TypedDict):
    """
    The structure to represent QnA predictions or ground truth.
    """

    # start and end are word indexes w.r.t context
    start: int
    end: int
    conf: float


def calculate_qa_predictions(
    tok_start_logits: np.array, tok_end_logits: np.array, doc_word_map: np.array
) -> QAPrediction:
    """
    Calculate the start and end positions and their confidences from start and
    end logits.
    """
    tok_start_logits = np.expand_dims(tok_start_logits, axis=0)
    start_pred = calculate_predictions(tok_start_logits)
    start_tok = start_pred["predicted_classes"][0]
    start_conf = start_pred["prediction_confidences"][0]

    if start_tok < len(tok_end_logits):
        tok_end_logits = tok_end_logits[start_tok:]
        tok_end_logits = np.expand_dims(tok_end_logits, axis=0)
        end_pred = calculate_predictions(tok_end_logits)
        end_tok = end_pred["predicted_classes"][0] + start_tok
    else:
        tok_end_logits = np.expand_dims(tok_end_logits, axis=0)
        end_pred = calculate_predictions(tok_end_logits)
        end_tok = end_pred["predicted_classes"][0]
    end_conf = end_pred["prediction_confidences"][0]

    conf = (start_conf + end_conf) / 2
    start_word = doc_word_map[start_tok]
    end_word = doc_word_map[end_tok] + 1
    return QAPrediction(start=start_word, end=end_word, conf=conf)


def extract_answer_words(doc: Dict[Text, Any], qa_pred: QAPrediction, is_gold=False) -> List[Dict[Text, Any]]:
    """
    Given a QA prediction, create a list of answer words.
    """
    words = []
    for idx in range(qa_pred["start"], qa_pred["end"]):
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
            conf=qa_pred["conf"],
            idx=idx,
        )
        if not is_gold:
            if doc.get("word_line_idx", None) is not None:
                word["word_line_idx"] = doc["word_line_idx"][idx]
            if doc.get("word_in_line_idx", None) is not None:
                word["word_in_line_idx"] = doc["word_in_line_idx"][idx]
        words.append(word)
    return words


def get_qa_predictions(predictions, dataset: Dataset, qchunk: Dict[Text, int]) -> Tuple[QAPrediction, QAPrediction]:
    """
    Dechunk the question chunks and get QA prediction from raw predictions.
    """
    start_logits, end_logits = predictions.predictions
    from_idx = qchunk["from_idx"]
    to_idx = qchunk["to_idx"]

    # Get the chunk which contains the answer otherwise get last/any chunk
    for idx in range(from_idx, to_idx):
        doc = dataset[from_idx]
        if doc["start_positions"] > 0:
            break

    # Gathering relevant feature chunks
    word_map_lst = dataset["word_map"][from_idx:to_idx]
    content_mask_lst = dataset["content_tokens_mask"][from_idx:to_idx]
    chunk_ranges = dataset["chunk_ranges"][from_idx:to_idx]
    start_logits_arr = start_logits[from_idx:to_idx]
    end_logits_arr = end_logits[from_idx:to_idx]

    # Dechunking
    doc_word_map = join_chunks(word_map_lst, chunk_ranges, None)
    tok_start_logits = join_chunks(start_logits_arr, chunk_ranges, content_mask_lst)
    tok_end_logits = join_chunks(end_logits_arr, chunk_ranges, content_mask_lst)

    # Calculate Predictions and goldens
    pred_ans = calculate_qa_predictions(tok_start_logits, tok_end_logits, doc_word_map)
    context_start = np.where(doc["content_tokens_mask"])[0][0]
    chunk_start = doc["chunk_ranges"][0]
    if doc["start_positions"] == doc["end_positions"] == 0:  # no answer
        start_word = end_word = 0
    else:
        # Convert positions w.r.t. context
        start_word = doc_word_map[doc["start_positions"] + chunk_start - context_start]
        end_word = doc_word_map[doc["end_positions"] + chunk_start - context_start] + 1
    gold_ans = QAPrediction(start=start_word, end=end_word, conf=0)
    return pred_ans, gold_ans


def _create_question_chunks(dataset: Dataset, chunk_from_idx: int, chunk_to_idx: int) -> List[Dict[Text, int]]:
    """
    Create question chunks from all the chunks of a document. Using chunk
    ranges to extract the question chunks.
    """
    # No. of chunks = chunk per doc x No. of questions
    doc_chunk_ranges = dataset["chunk_ranges"][chunk_from_idx:chunk_to_idx]
    question_chunk = dict(from_idx=chunk_from_idx, to_idx=chunk_from_idx + 1)
    for idx, chunk in enumerate(doc_chunk_ranges[1:], start=1):
        chunk_id = chunk_from_idx + idx
        if chunk[0] > 0:
            question_chunk["to_idx"] = chunk_id + 1
        else:
            yield question_chunk
            question_chunk = dict(from_idx=chunk_id, to_idx=chunk_id + 1)
    yield question_chunk


def get_predictions_for_qa(predictions: Tuple, dataset: Dataset) -> Dict[Text, Dict]:
    """
    Extract prediction dictionary from raw predictions for single QnA task.
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

        predicted_entity_words = {}
        gold_entity_words = {}
        # Iterate through question chunks.
        for qchunk in _create_question_chunks(dataset, chunk_from_idx, chunk_to_idx):
            doc = dataset[qchunk["from_idx"]]
            question = doc["entities"]["name"][0]
            pred_ans, gold_ans = get_qa_predictions(predictions, dataset, qchunk)
            predicted_entity_words[question] = extract_answer_words(doc, pred_ans)
            gold_entity_words[question] = extract_answer_words(doc, gold_ans, is_gold=True)
            logging.debug(f"{question} \n{pred_ans} \n{gold_ans}")

        doc_dict = create_entities(predicted_entity_words, gold_entity_words, label_list)
        is_test_file = doc["is_test_file"] if "is_test_file" in doc else False
        pred_dict[doc["id"]] = {"is_test_file": is_test_file, "entities": doc_dict}
    return pred_dict
