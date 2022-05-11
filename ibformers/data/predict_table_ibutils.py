from typing import List, Dict

import numpy as np

from instabase.dataset_utils.shared_types import PredictionFieldDict, RowPredictionInstanceDict, \
    ColumnPredictionInstanceDict, CellPredictionInstanceDict, IndexedWordDict, TableExtractionPredictionInstanceDict, \
    BboxDict


def convert_tables_to_pred_field(label: str, tables: List[Dict]) -> PredictionFieldDict:
    out_tabs = []
    for table in tables:
        rows = [
            RowPredictionInstanceDict(
                bbox=get_backend_bbox(r["bbox"]),
                page_index=r["page_idx"],
                label_id="",
                is_header=False,
                confidence=r["confidence_score"],
            )
            for r in table["rows"]
        ]
        cols = [
            ColumnPredictionInstanceDict(
                bboxes=[get_backend_bbox(r["bbox"])],
                page_indexes=[r["page_idx"]],
                label_id="",
                is_header=False,
                confidence=r["confidence_score"],
            )
            for r in table["columns"]
        ]
        cells = [
            CellPredictionInstanceDict(
                bbox=get_backend_bbox(r["bbox"]),
                page_index=r["page_idx"],
                row_start_index=r["row_start_index"],
                row_end_index=r["row_end_index"],
                col_start_index=r["col_start_index"],
                col_end_index=r["col_end_index"],
                words=[IndexedWordDict(line_index=w[0], word_index=w[1]) for w in r["indexed_words"]],
            )
            for r in table["cells"]
        ]

        avg_conf_str = np.mean([r["confidence"] for r in rows + cols])
        avg_conf_tab = (table["confidence_score"] + avg_conf_str) / 2

        tab_predictions = TableExtractionPredictionInstanceDict(
            bboxes=[get_backend_bbox(bb) for bb in table["table_bboxes"]],
            page_indexes=list(range(table["start_page_index"], table["end_page_index"] + 1)),
            rows=rows,
            cols=cols,
            cells=cells,
            avg_confidence=avg_conf_tab,
        )

        # do not output shitty tables
        if table["confidence_score"] > 0.3:
            out_tabs.append(tab_predictions)

    field_name = label if label != "O" else None
    field = PredictionFieldDict(field_name=field_name, field_type="Table", table_annotations=out_tabs)
    return field


def get_backend_bbox(bb: List) -> BboxDict:
    return BboxDict(top_x=bb[0], top_y=bb[1], bottom_x=bb[2], bottom_y=bb[3])
