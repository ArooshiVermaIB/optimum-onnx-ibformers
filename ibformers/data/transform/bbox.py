from typing import List, Tuple, Dict
import numpy as np
from ibformers.data.utils import feed_single_example


def _fix_for_negative_dims(bboxes: np.ndarray) -> np.ndarray:
    """
    fix the bboxes which do not meet conditions x1<x2 and y1<y2 for bbox format (x1,y1,x2,y2)
    """
    fix_bboxes = np.stack(
        (bboxes[:, [0, 2]].min(-1), bboxes[:, [1, 3]].min(-1), bboxes[:, [0, 2]].max(-1), bboxes[:, [1, 3]].max(-1)),
        axis=1,
    )
    return fix_bboxes


@feed_single_example
def norm_bboxes_for_layoutlm(example: Dict, **kwargs):
    bboxes, page_bboxes, page_spans = (
        example["bboxes"],
        example["page_bboxes"],
        example["page_spans"],
    )
    norm_bboxes, norm_page_bboxes = _norm_bboxes_for_layoutlm(bboxes, page_bboxes, page_spans)
    # layoutlm expect bboxes to be in format x1,y1,x2,y2 where x1<x2 and y1<y2
    fixed_bboxes = _fix_for_negative_dims(norm_bboxes)
    min_val = fixed_bboxes.min()
    max_val = fixed_bboxes.max()

    if min_val < 0 or max_val > 1000:
        ex = fixed_bboxes.max(-1).argmax() if max_val > 1000 else fixed_bboxes.min(-1).argmin()
        ex_bbox = bboxes[ex]
        raise ValueError(
            f"Bboxes are outside of required range 0-1000. Range: {min_val} - {max_val} "
            f"Example Bbox: {ex_bbox}, Page bbox: {page_bboxes}, Page Spans: {page_spans}"
        )

    return {"bboxes": fixed_bboxes, "page_bboxes": norm_page_bboxes}


def _norm_bboxes_for_layoutlm(
    bboxes: np.ndarray, page_bboxes: np.ndarray, page_spans: List[Tuple[int, int]]
) -> Tuple[np.ndarray, np.ndarray]:
    page_bboxes = np.array(page_bboxes, dtype=np.int64)
    for (_, _, _, page_height), (page_start_i, page_end_i) in zip(page_bboxes, page_spans):
        bboxes[page_start_i:page_end_i, [1, 3]] = bboxes[page_start_i:page_end_i, [1, 3]] / (page_height / 1000 + 1e-10)
    page_bboxes[:, 3] = 1000

    return bboxes, page_bboxes
