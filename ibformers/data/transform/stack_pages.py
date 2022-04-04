from typing import Dict

import numpy as np

from ibformers.data.utils import feed_single_example


@feed_single_example
def stack_pages(example: Dict, **kwargs):
    bboxes = np.array(example["bboxes"])
    if example["token_page_nums"][0] == example["token_page_nums"][-1]:
        return {"bboxes": bboxes}
    page_nums = np.array(example["token_page_nums"])
    content_mask = np.array(example["content_tokens_mask"])

    # stack pages one under each other and squeeze them so the dimension is not greater than 1000
    page_offset = page_nums - page_nums[0]
    y_coord = bboxes[content_mask][:, (1, 3)] + page_offset[:, None] * 1000
    y_coord_norm = y_coord / (page_offset[-1] + 1)
    bboxes[content_mask, 1] = y_coord_norm[:, 0]
    bboxes[content_mask, 3] = y_coord_norm[:, 1]

    return {"bboxes": bboxes}
