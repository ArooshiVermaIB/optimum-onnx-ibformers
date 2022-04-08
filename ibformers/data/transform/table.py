from typing import Dict, List, Any, Callable, Tuple

import numpy as np

from ibformers.data.utils import (
    feed_single_example,
    feed_single_example_and_flatten,
    convert_to_list_of_dicts,
    convert_to_dict_of_lists,
)
from ibformers.datasets.table_utils import (
    DetrStructureClassNames,
    DETR_DETECTION_CLASS_MAP,
    DetrDetectionClassNames,
    DETR_STRUCTURE_CLASS_MAP,
)

BASE_MARGIN = 50


@feed_single_example
def calculate_margins(example: Dict, **kwargs):
    scale_factors = [
        bbox[-1] / orig_bbox[-1] for bbox, orig_bbox in zip(example["page_bboxes"], example["page_original_bboxes"])
    ]
    return {"margins": [int(BASE_MARGIN * scale_factor) for scale_factor in scale_factors]}


@feed_single_example_and_flatten
def prepare_image_and_table_data(example: Dict, **kwargs):
    list_of_tables = convert_to_list_of_dicts(example["tables"])

    for page_no in np.unique(example["word_page_nums"]):
        tables_within_page = [table for table in list_of_tables if is_table_in_page(table, page_no)]

        # getting some negative examples. Not sure if necessary though
        if len(tables_within_page) == 0:
            yield get_empty_table_anno(example, page_no)
            continue

        all_page_table_objects = [extract_page_table_objects(table, page_no) for table in tables_within_page]
        detection_labels = prepare_detection_labels(all_page_table_objects)

        for page_table_objects in all_page_table_objects:
            table_subimage_bbox, bbox_shift_vector, margin = prepare_table_subimage(
                example, page_table_objects, page_no
            )
            structure_labels = prepare_structure_labels(page_table_objects, bbox_shift_vector)
            yield {
                **example,
                "table_page_bbox": example["page_bboxes"][page_no],
                "table_page_no": page_no,
                "structure_image_bbox": table_subimage_bbox,
                "page_margins": margin,
                **detection_labels,
                **structure_labels,
            }


def is_table_in_page(table, page_no):
    return table["start_page_index"] <= page_no <= table["end_page_index"]


def get_empty_table_anno(example, page_no):
    return {
        **example,
        "table_page_bbox": example["page_bboxes"][page_no],
        "table_page_no": page_no,
        "structure_image_bbox": example["page_bboxes"][page_no],
        "page_margins": example["margins"][page_no],
        "raw_detection_boxes": np.empty((0, 4)),
        "detection_labels": np.empty((0,), dtype=np.int32),
        "raw_structure_boxes": np.empty((0, 4)),
        "structure_labels": np.empty((0,), dtype=np.int32),
        "bbox_shift_vector": [0, 0, 0, 0],
    }


def extract_page_table_objects(table: Dict[str, Any], page_no: int) -> Dict[str, Any]:
    page_compare_fn = lambda x: x["page_idx"] == page_no
    return {
        "table_bbox": table["table_bboxes"][page_no],
        "rows": _filter_dict_items(table["rows"], page_compare_fn),
        "columns": _filter_dict_items(table["columns"], page_compare_fn),
        "cells": _filter_dict_items(table["cells"], page_compare_fn),
    }


def _filter_dict_items(dict_of_lists: Dict[str, List[Any]], filter_fn: Callable[[Any], bool]) -> Dict[str, List[Any]]:
    list_of_dicts = convert_to_list_of_dicts(dict_of_lists)
    filtered = list(filter(filter_fn, list_of_dicts))
    return convert_to_dict_of_lists(filtered, dict_of_lists.keys())


def prepare_detection_labels(all_page_table_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    table_bboxes = [table["table_bbox"] for table in all_page_table_objects]
    return {
        "raw_detection_boxes": np.array(table_bboxes),
        "detection_labels": [DETR_DETECTION_CLASS_MAP[DetrDetectionClassNames.TABLE]] * len(table_bboxes),
    }


def get_spanning_cells(page_table_objects: Dict[str, Any]) -> Dict[str, Any]:
    spanning_cell_fn = lambda cell: (
        (cell["row_start_index"] != cell["row_end_index"]) or (cell["col_start_index"] != cell["col_end_index"])
    )
    return _filter_dict_items(page_table_objects["cells"], spanning_cell_fn)


def split_rows_and_headers(page_table_objects: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rows = _filter_dict_items(page_table_objects["rows"], lambda x: not x["is_header"])
    column_headers = _filter_dict_items(page_table_objects["rows"], lambda x: x["is_header"])
    return rows, column_headers


def prepare_structure_labels(page_table_objects: Dict[str, Any], bbox_shift_vector: np.array) -> Dict[str, Any]:
    table_bboxes = [page_table_objects["table_bbox"]]
    spanning_cells = get_spanning_cells(page_table_objects)
    rows, column_headers = split_rows_and_headers(page_table_objects)
    labels = (
        [DETR_STRUCTURE_CLASS_MAP[DetrStructureClassNames.ROW]] * len(rows["bbox"])
        + [DETR_STRUCTURE_CLASS_MAP[DetrStructureClassNames.COLUMN_HEADER]] * len(column_headers["bbox"])
        + [DETR_STRUCTURE_CLASS_MAP[DetrStructureClassNames.COLUMN]] * len(page_table_objects["columns"]["bbox"])
        + [DETR_STRUCTURE_CLASS_MAP[DetrStructureClassNames.SPANNING_CELL]] * len(spanning_cells["bbox"])
        + [DETR_STRUCTURE_CLASS_MAP[DetrStructureClassNames.TABLE]]
    )
    bboxes = np.array(
        rows["bbox"]
        + column_headers["bbox"]
        + page_table_objects["columns"]["bbox"]
        + spanning_cells["bbox"]
        + table_bboxes
    )
    bboxes += bbox_shift_vector

    return {
        "raw_structure_boxes": bboxes,
        "structure_labels": labels,
        "bbox_shift_vector": bbox_shift_vector,
    }


def prepare_table_subimage(example, table, page_no):
    margin = example["margins"][page_no]
    subimage_bbox = np.array(table["table_bbox"]) + np.array([-margin, -margin, margin, margin])
    subimage_bbox = subimage_bbox.clip(min=0)
    bbox_shift_vector = -np.array([subimage_bbox[0], subimage_bbox[1], subimage_bbox[0], subimage_bbox[1]])
    return subimage_bbox, bbox_shift_vector, margin


@feed_single_example
def normalize_object_bboxes(example: Dict, **kwargs):
    detection_normalize_factor = np.array(example["page_bboxes"][0])[[2, 3, 2, 3]]
    structure_image_size = example["structure_image_bbox"][[2, 3]] - example["structure_image_bbox"][[0, 1]]
    structure_normalize_factor = np.array([*structure_image_size, *structure_image_size])
    detection_boxes = example["raw_detection_boxes"]
    structure_boxes = example["raw_structure_boxes"]
    norm_detection_boxes = detection_boxes / detection_normalize_factor
    norm_structure_boxes = structure_boxes / structure_normalize_factor
    return {
        "structure_boxes": norm_structure_boxes,
        "detection_boxes": norm_detection_boxes,
        "structure_image_size": [structure_image_size[1], structure_image_size[0]] * 2,
    }


@feed_single_example
def convert_bboxes_to_center_based(example: Dict, **kwargs):
    # xxyy -> cx, cy, h, w
    detection_boxes = example["detection_boxes"]
    structure_boxes = example["structure_boxes"]
    centered_detection_boxes = _convert_array_to_center_based(detection_boxes)
    centered_structure_boxes = _convert_array_to_center_based(structure_boxes)
    return {
        "structure_boxes": centered_structure_boxes,
        "detection_boxes": centered_detection_boxes,
    }


def _convert_array_to_center_based(array: np.array):
    return np.stack(
        [
            (array[:, 2] + array[:, 0]) / 2,
            (array[:, 3] + array[:, 1]) / 2,
            (array[:, 2] - array[:, 0]),
            (array[:, 3] - array[:, 1]),
        ],
        axis=1,
    )
