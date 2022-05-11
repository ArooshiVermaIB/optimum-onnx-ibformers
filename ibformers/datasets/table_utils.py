from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any


class DetrStructureClassNames(Enum):
    TABLE = "table"
    COLUMN = "table column"
    ROW = "table row"
    COLUMN_HEADER = "table column header"
    PROJECTED_ROW_HEADER = "table projected row header"
    SPANNING_CELL = "table spanning cell"
    NO_OBJECT = "no object"


DETR_STRUCTURE_CLASS_MAP = {k: v for v, k in enumerate(DetrStructureClassNames)}
DETR_STRUCTURE_CLASSNAME_MAP = {k.value: v for v, k in enumerate(DetrStructureClassNames)}


DETR_STRUCTURE_CLASS_THRESHOLDS = {
    DetrStructureClassNames.TABLE.value: 0.5,
    DetrStructureClassNames.COLUMN.value: 0.5,
    DetrStructureClassNames.ROW.value: 0.5,
    DetrStructureClassNames.COLUMN_HEADER.value: 0.5,
    DetrStructureClassNames.PROJECTED_ROW_HEADER.value: 0.5,
    DetrStructureClassNames.SPANNING_CELL.value: 20,  # TODO: bring back when spanning cells are supported in ML Studio
    DetrStructureClassNames.NO_OBJECT.value: 20,
}


class DetrDetectionClassNames(Enum):
    TABLE = "table"
    TABLE_ROTATED = "table rotated"
    NO_OBJECT = "no object"


DETR_DETECTION_CLASS_MAP = {k: v for v, k in enumerate(DetrDetectionClassNames)}
DETR_DETECTION_CLASS_THRESHOLDS = {
    DetrDetectionClassNames.TABLE: 0.5,
    DetrDetectionClassNames.TABLE_ROTATED: 0.5,
    DetrDetectionClassNames.NO_OBJECT: 20,
}


@dataclass
class CellAnnotation:
    row_start_index: int
    row_end_index: int
    col_start_index: int
    col_end_index: int
    page_idx: int
    bbox: Tuple[float, float, float, float]
    is_column_header: Optional[bool] = None
    is_row_header: Optional[bool] = None

    def to_json(self):
        return {
            "row_start_index": self.row_start_index,
            "row_end_index": self.row_end_index,
            "col_start_index": self.col_start_index,
            "col_end_index": self.col_end_index,
            "page_idx": self.page_idx,
            "bbox": self.bbox,
            "is_column_header": self.is_column_header,
            "is_row_header": self.is_row_header,
        }


@dataclass
class CellPrediction(CellAnnotation):
    indexed_words: List[Tuple[int, int]] = None

    def to_json(self):
        base_json = super().to_json()
        base_json["indexed_words"] = [] if self.indexed_words is None else [list(t) for t in self.indexed_words]
        return base_json


@dataclass
class RowAnnotation:
    bbox: Tuple[float, float, float, float]
    page_idx: int
    is_header: Optional[bool] = False
    label_id: Optional[int] = None
    confidence_score: Optional[float] = None

    def to_json(self):
        if self.confidence_score is not None:
            return {
                "bbox": self.bbox,
                "page_idx": self.page_idx,
                "is_header": self.is_header,
                "label_id": self.label_id,
                "confidence_score": self.confidence_score,
            }
        else:
            return {
                "bbox": self.bbox,
                "page_idx": self.page_idx,
                "is_header": self.is_header,
                "label_id": self.label_id,
            }


@dataclass
class ColumnAnnotation:
    bbox: Tuple[float, float, float, float]
    page_idx: int
    is_header: Optional[bool] = False
    label_id: Optional[int] = None
    confidence_score: Optional[float] = None

    def to_json(self):
        if self.confidence_score is not None:
            return {
                "bbox": self.bbox,
                "page_idx": self.page_idx,
                "is_header": self.is_header,
                "label_id": self.label_id,
                "confidence_score": self.confidence_score,
            }
        else:
            return {
                "bbox": self.bbox,
                "page_idx": self.page_idx,
                "is_header": self.is_header,
                "label_id": self.label_id,
            }


@dataclass
class TableAnnotation:
    table_idx: int
    start_page_index: int
    end_page_index: int
    rows: List[RowAnnotation]
    columns: List[ColumnAnnotation]
    cells: List[CellAnnotation]
    table_bboxes: List[Tuple[float, float, float, float]]
    base_page_bboxes: Optional[List[Tuple[float, float, float, float]]] = None
    table_label_id: Optional[int] = None
    table_label_name: Optional[str] = None
    confidence_score: Optional[float] = None

    def to_json(self):
        return {
            "table_idx": self.table_idx,
            "start_page_index": self.start_page_index,
            "end_page_index": self.end_page_index,
            "rows": [r.to_json() for r in self.rows],
            "columns": [r.to_json() for r in self.columns],
            "cells": [r.to_json() for r in self.cells],
            "table_bboxes": self.table_bboxes,
            "base_page_bboxes": self.base_page_bboxes,
            "table_label_id": self.table_label_id,
            "table_label_name": self.table_label_name,
            "confidence_score": self.confidence_score,
        }


@dataclass
class DocTableAnnotations:
    anno_id: str
    table_annotations: List[TableAnnotation]

    def __post_init__(self):
        for table_anno in self.table_annotations:
            assert table_anno.base_page_bboxes == self.table_annotations[0].base_page_bboxes

    def to_dataset_features_part(self) -> Dict[str, Any]:
        dataset_features = {
            "page_bboxes": self.table_annotations[0].base_page_bboxes,
            "id": self.anno_id,
            "tables": [asdict(table) for table in self.table_annotations],
        }
        for table_anno_dict in dataset_features["tables"]:
            table_anno_dict.pop("base_page_bboxes")
        return dataset_features
