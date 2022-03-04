from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class CellAnnotation:
    row_ids: List[int]
    col_ids: List[int]
    page_idx: int
    bbox: Tuple[float, float, float, float]
    is_column_header: Optional[bool] = None
    is_row_header: Optional[bool] = None


@dataclass
class RowAnnotation:
    bbox: Tuple[float, float, float, float]
    page_idx: int
    is_column_header: Optional[bool] = None


@dataclass
class ColumnAnnotation:
    bbox: Tuple[float, float, float, float]
    page_idx: int


@dataclass
class TableAnnotation:
    table_idx: int
    page_span: Tuple[int, int]
    rows: List[RowAnnotation]
    columns: List[ColumnAnnotation]
    cells: List[CellAnnotation]
    table_bboxes: List[Tuple[float, float, float, float]]
    base_page_bboxes: List[Tuple[float, float, float, float]]


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
