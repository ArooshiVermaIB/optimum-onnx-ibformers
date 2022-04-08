from typing import Tuple, Optional, List, Iterator, Dict, Any

import numpy as np
from datasets import Dataset

from ibformers.data.utils import convert_to_list_of_dicts, convert_to_dict_of_lists
from ibformers.datasets.table_utils import (
    DETR_STRUCTURE_CLASS_THRESHOLDS,
    DETR_STRUCTURE_CLASSNAME_MAP,
    TableAnnotation,
    RowAnnotation,
    ColumnAnnotation,
    CellPrediction,
)
from ibformers.third_party.grits.grits import objects_to_cells, compute_metrics


def doc_page_iter(doc_ids: List[str], page_numbers: List[int]) -> Iterator[Tuple[str, int, int, int]]:
    """
    Get an iterator of dataset index ranges that group together examples from the same document and page.
    :param doc_ids: list of document ids
    :param page_numbers: list of page numbers
    :return: Iterator of tuples
    """
    from_idx = 0
    next_doc_ids = doc_ids[1:] + ["*end*"]
    next_page_numbers = page_numbers[1:] + [-1]
    for i, (doc_id, next_doc_id, page_no, next_page_no) in enumerate(
        zip(doc_ids, next_doc_ids, page_numbers, next_page_numbers)
    ):
        if doc_id != next_doc_id or page_no != next_page_no:
            yield doc_id, page_no, from_idx, i + 1
            from_idx = i + 1


def get_predictions_for_table_detr(predictions: Tuple, dataset: Dataset, label_list: Optional[List] = None):
    model_predictions, labels = predictions
    assert len(dataset) == len(model_predictions) == len(labels)
    ids = dataset["id"]
    page_numbers = dataset["table_page_no"]
    predictions = {}
    metrics = []
    for doc_id, page_no, chunk_from_idx, chunk_to_idx in doc_page_iter(ids, page_numbers):
        valid_example_dict = dataset[chunk_from_idx:chunk_to_idx]
        valid_examples = convert_to_list_of_dicts(valid_example_dict)
        valid_predictions = model_predictions[chunk_from_idx:chunk_to_idx]
        valid_labels = labels[chunk_from_idx:chunk_to_idx]
        page_annotations, page_metrics = process_document_page(
            valid_examples, valid_predictions, valid_labels, doc_id, page_no
        )
        predictions[doc_id] = [p.to_json() for p in page_annotations]
        metrics.extend(page_metrics)
    metrics = convert_to_dict_of_lists(metrics, metrics[0].keys()) if len(metrics) > 0 else []
    metrics = {k: np.mean(v) for k, v in metrics.items()}
    return {"metrics": metrics, "predictions": predictions}


def process_document_page(
    valid_examples, valid_predictions, valid_labels, doc_id, page_no
) -> Tuple[List[TableAnnotation], List[Dict]]:
    page_tokens = get_page_tokens(valid_examples[0], page_no)
    page_table_annos: List[TableAnnotation] = []
    page_metrics = []
    for i, (example, prediction, label) in enumerate(zip(valid_examples, valid_predictions, valid_labels)):
        structure_predictions = prediction[1]
        for structure_prediction in structure_predictions:
            pred_boxes = structure_prediction[1] - np.array(example["bbox_shift_vector"])
            pred_boxes = [o.tolist() for o in pred_boxes]
            pred_labels = structure_prediction[2].tolist()
            pred_scores = structure_prediction[0].tolist()
            pred_table_structures, pred_cells, pred_confidence_score, table_bbox = objects_to_cells(
                bboxes=pred_boxes,
                labels=pred_labels,
                scores=pred_scores,
                page_tokens=page_tokens,
                structure_class_names=list(DETR_STRUCTURE_CLASSNAME_MAP.keys()),
                structure_class_thresholds=DETR_STRUCTURE_CLASS_THRESHOLDS,
                structure_class_map=DETR_STRUCTURE_CLASSNAME_MAP,
            )

            gold_bboxes = list_of_bboxes_to_array(example["raw_structure_boxes"]) - np.array(
                example["bbox_shift_vector"]
            )
            gold_bboxes = [o.tolist() for o in gold_bboxes]
            gold_labels = example["structure_labels"]
            gold_scores = [1] * len(example["raw_structure_boxes"])
            gold_table_structures, gold_cells, gold_confidence_score, _ = objects_to_cells(
                bboxes=gold_bboxes,
                labels=gold_labels,
                scores=gold_scores,
                page_tokens=page_tokens,
                structure_class_names=list(DETR_STRUCTURE_CLASSNAME_MAP.keys()),
                structure_class_thresholds=DETR_STRUCTURE_CLASS_THRESHOLDS,
                structure_class_map=DETR_STRUCTURE_CLASSNAME_MAP,
            )

            metrics = compute_metrics(
                gold_bboxes, gold_labels, gold_scores, gold_cells, pred_boxes, pred_labels, pred_scores, pred_cells
            )

            scaled_table_structures, scaled_cells, scaled_table_bbox = get_scaled_table_contents(
                pred_table_structures, pred_cells, table_bbox, example, page_no
            )
            predicted_table_annotation = table_structures_to_table_annotation(
                scaled_table_structures, scaled_cells, scaled_table_bbox, example, page_no, i
            )
            page_table_annos.append(predicted_table_annotation)

            page_metrics.append(metrics)
    return page_table_annos, page_metrics


def get_scaled_table_contents(pred_table_structures, pred_cells, table_bbox, example, page_no):
    scale_factor = example["page_original_bboxes"][page_no][-1] / example["table_page_bbox"][-1]
    scaled_table_structures = {}
    for structure_name, structures in pred_table_structures.items():
        scaled_table_structures[structure_name] = [
            get_structure_with_scaled_bbox(structure, scale_factor) for structure in structures
        ]
    scaled_cells = [get_cell_with_scaled_bbox(cell, scale_factor) for cell in pred_cells]
    scaled_table_bbox = [coord * scale_factor for coord in table_bbox]

    return scaled_table_structures, scaled_cells, scaled_table_bbox


def get_structure_with_scaled_bbox(structure: Dict[str, Any], bbox_scale_factor: float):
    structure = structure.copy()
    structure["bbox"] = [coord * bbox_scale_factor for coord in structure["bbox"]]
    return structure


def get_cell_with_scaled_bbox(cell: Dict[str, Any], bbox_scale_factor: float):
    cell = cell.copy()
    cell["bbox"] = [coord * bbox_scale_factor for coord in cell["bbox"]]
    cell["spans"] = [get_structure_with_scaled_bbox(span, bbox_scale_factor) for span in cell["spans"]]
    return cell


def list_of_bboxes_to_array(bbox_list):
    return np.array(bbox_list).reshape((len(bbox_list), 4))


def get_page_tokens(example, page_no):
    boxes = np.array(example["bboxes"])
    words = np.array(example["words"])
    word_line_idxs = np.array(example["word_line_idx"])
    word_in_line_idxs = np.array(example["word_in_line_idx"])
    word_page_nums = np.array(example["word_page_nums"])
    valid_boxes = boxes[word_page_nums == page_no]
    valid_words = words[word_page_nums == page_no]
    return [
        {
            "bbox": bbox.tolist(),
            "text": text,
            "flags": 0,
            "span_num": i,
            "line_num": word_line,
            "block_num": 0,
            "word_in_line": word_in_line,
        }
        for i, (bbox, text, word_line, word_in_line) in enumerate(
            zip(valid_boxes, valid_words, word_line_idxs, word_in_line_idxs)
        )
    ]


def table_structures_to_table_annotation(
    pred_table_structures, pred_cells, table_bbox, example, page_no, table_idx
) -> TableAnnotation:
    row_annotations = make_row_annotations(pred_table_structures["rows"], page_no)
    column_annotations = make_column_annotations(pred_table_structures["columns"], page_no)
    cell_annotations = make_cell_predicitons(pred_cells, page_no)

    table_annotation = TableAnnotation(
        table_idx=table_idx,
        start_page_index=page_no,
        end_page_index=page_no,
        rows=row_annotations,
        columns=column_annotations,
        cells=cell_annotations,
        table_bboxes=[table_bbox],
    )
    return table_annotation


def make_row_annotations(row_structures, page_no):
    return [
        RowAnnotation(bbox=struct["bbox"], page_idx=page_no, is_header=struct["header"]) for struct in row_structures
    ]


def make_column_annotations(column_structures, page_no):
    return [
        ColumnAnnotation(
            bbox=struct["bbox"],
            page_idx=page_no,
        )
        for struct in column_structures
    ]


def make_cell_predicitons(cells, page_no):
    return [
        CellPrediction(
            row_start_index=min(cell["row_nums"]),
            row_end_index=max(cell["row_nums"]),
            col_start_index=min(cell["column_nums"]),
            col_end_index=max(cell["column_nums"]),
            page_idx=page_no,
            bbox=cell["bbox"],
            indexed_words=make_indexed_words(cell),
        )
        for cell in cells
    ]


def make_indexed_words(cell):
    return [(cell_span["line_num"], cell_span["word_in_line"]) for cell_span in cell["spans"]]
