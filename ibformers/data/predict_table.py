from typing import Tuple, List, Iterator, Dict, Any
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


def doc_page_iter(doc_ids: List[str], page_numbers: List[int]) -> Iterator[Tuple[str, int, int, int, int]]:
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
            yield doc_id, page_no, from_idx, i + 1, i
            from_idx = i + 1


def get_predictions_for_table_detr(predictions: Tuple, dataset: Dataset):
    label_list = dataset.features["tables"].feature["table_label_id"].names

    model_predictions, labels = predictions
    assert len(dataset) == len(model_predictions)
    if len(labels) > 0:
        assert len(model_predictions) == len(labels)
    else:
        # create some dummy labels
        labels = [None] * len(model_predictions)
    ids = dataset["id"]
    page_numbers = dataset["table_page_no"]
    are_test_files = dataset["is_test_file"]
    predictions = {}
    metrics = []
    for doc_id, page_no, chunk_from_idx, chunk_to_idx, idx in doc_page_iter(ids, page_numbers):
        valid_example_dict = dataset[chunk_from_idx:chunk_to_idx]
        valid_examples = convert_to_list_of_dicts(valid_example_dict)
        valid_predictions = model_predictions[chunk_from_idx:chunk_to_idx]
        valid_labels = labels[chunk_from_idx:chunk_to_idx]
        page_annotations, page_metrics = process_document_page(
            valid_examples, valid_predictions, valid_labels, doc_id, page_no, label_list
        )
        ents = [p.to_json() for p in page_annotations]
        ents_per_label = {lab: [e for e in ents if e["table_label_name"] == lab] for lab in label_list}
        predictions[doc_id] = {"is_test_file": are_test_files[idx], "entities": ents_per_label}
        metrics.extend(page_metrics)
    metrics = convert_to_dict_of_lists(metrics, metrics[0].keys()) if len(metrics) > 0 else {}
    metrics = {k: np.mean(np.nan_to_num(v)) for k, v in metrics.items()}
    return {"metrics": metrics, "predictions": predictions}


def process_document_page(
    valid_examples, valid_predictions, valid_labels, doc_id, page_no, label_list
) -> Tuple[List[TableAnnotation], List[Dict]]:
    page_tokens = get_page_tokens(valid_examples[0], page_no)
    page_table_annos: List[TableAnnotation] = []
    page_metrics = []
    for i, (example, prediction, label) in enumerate(zip(valid_examples, valid_predictions, valid_labels)):
        record_page_no = example["record_table_page_no"]
        detect_predictions, structure_predictions = prediction
        for structure_prediction in structure_predictions:
            pred_boxes = structure_prediction[1]
            pred_boxes = [o.tolist() for o in pred_boxes]
            pred_labels = structure_prediction[2].tolist()
            pred_scores = structure_prediction[0].tolist()
            pred_label_id = int(structure_prediction[3]) if len(structure_prediction) == 4 else 0
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

            expanded_raw_structures, expanded_raw_cells = expand_table_contents(
                pred_table_structures, pred_cells, table_bbox
            )
            scaled_table_structures, scaled_cells, scaled_table_bbox = get_scaled_table_contents(
                pred_table_structures, pred_cells, table_bbox, example, page_no
            )
            expanded_table_structures, expanded_cells = expand_table_contents(
                scaled_table_structures, scaled_cells, scaled_table_bbox
            )
            predicted_table_annotation = table_structures_to_table_annotation(
                expanded_table_structures,
                expanded_cells,
                scaled_table_bbox,
                example,
                record_page_no,
                i,
                pred_label_id,
                label_list,
                pred_confidence_score,
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


def expand_table_contents(table_structures, cells, table_bbox):
    expanded_structures = {}
    expanded_structures["columns"] = expand_structure_to_table(table_structures["columns"], table_bbox, True)
    expanded_structures["rows"] = expand_structure_to_table(table_structures["rows"], table_bbox, False)
    # we skip headers end supercells as they're not needed anymore
    expanded_cells = update_cell_bboxes(cells, expanded_structures)
    return expanded_structures, expanded_cells


def expand_structure_to_table(table_structures, table_bbox, is_col):
    total_structures = len(table_structures)
    low_to_avg_idx, high_to_avg_idx = (0, 2) if is_col else (1, 3)
    low_from_table_idx, high_from_table_idx = (1, 3) if is_col else (0, 2)
    low_coords = np.array([struct["bbox"][low_to_avg_idx] for struct in table_structures] + [0])
    high_coords = np.array([0] + [struct["bbox"][high_to_avg_idx] for struct in table_structures])
    average_coords = (high_coords + low_coords) / 2

    new_structures = []
    for i, structure in enumerate(table_structures):
        new_bbox = structure["bbox"].copy()
        new_bbox[low_to_avg_idx] = table_bbox[low_to_avg_idx] if i == 0 else average_coords[i]
        new_bbox[high_to_avg_idx] = table_bbox[high_to_avg_idx] if i == total_structures - 1 else average_coords[i + 1]
        new_bbox[low_from_table_idx] = table_bbox[low_from_table_idx]
        new_bbox[high_from_table_idx] = table_bbox[high_from_table_idx]
        new_structures.append({**structure, "bbox": new_bbox})
    return new_structures


def update_cell_bboxes(cells, table_structures):
    new_cells = []
    for cell in cells:
        cell_row_lower, cell_row_upper = cell["row_nums"][0], cell["row_nums"][-1]
        cell_col_lower, cell_col_upper = cell["column_nums"][0], cell["column_nums"][-1]
        new_bbox = [
            table_structures["columns"][cell_col_lower]["bbox"][0],
            table_structures["rows"][cell_row_lower]["bbox"][1],
            table_structures["columns"][cell_col_upper]["bbox"][2],
            table_structures["rows"][cell_row_upper]["bbox"][3],
        ]
        new_cells.append({**cell, "bbox": new_bbox})
    return new_cells


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
    pred_table_structures, pred_cells, table_bbox, example, page_no, table_idx, label_id, label_names, conf_score
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
        table_label_id=label_id,
        table_label_name=label_names[label_id],
        confidence_score=conf_score,
    )
    return table_annotation


def make_row_annotations(row_structures, page_no):
    return [
        RowAnnotation(
            bbox=struct["bbox"], page_idx=page_no, is_header=struct["header"], confidence_score=struct["score"]
        )
        for struct in row_structures
    ]


def make_column_annotations(column_structures, page_no):
    return [
        ColumnAnnotation(bbox=struct["bbox"], page_idx=page_no, confidence_score=struct["score"])
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
