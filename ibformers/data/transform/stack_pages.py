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


@feed_single_example
def filter_npages(example: Dict, npages_to_filter: int = -1, **kwargs):

    if npages_to_filter == -1:
        return {}

    page_nums = example["word_page_nums"]

    min_page = min(page_nums)
    max_page = min(max(page_nums), min_page + npages_to_filter - 1)

    page_spans = example["page_spans"]

    start_idx = page_spans[min_page][0]
    end_idx = page_spans[max_page][1]

    filter_slice = slice(start_idx, end_idx)

    keys_to_slice = ["words", "bboxes", "word_original_bboxes", "word_page_nums", "word_line_idx", "word_in_line_idx"]

    return {key: example[key][filter_slice] for key in example if key in keys_to_slice}
