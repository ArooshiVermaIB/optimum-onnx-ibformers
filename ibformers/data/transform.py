from typing import List, TypeVar, Tuple

import numpy as np
from typing_extensions import TypedDict

from ibformers.data.utils import convert_to_dict_of_lists, tag_answer_in_doc, feed_single_example


@feed_single_example
def fuzzy_tag_in_document(example, **kwargs):
    # try to find an answer inside the text of the document
    # example will be skipped in case of no spans found
    answers = tag_answer_in_doc(words=example['words'], answer=example['answer'])
    if len(answers) == 0:
        return None
    else:
        return {'detected_answers': answers}


@feed_single_example
def add_token_labels_qa(example, **kwargs):
    token_starts = example["token_starts"]
    answers = example["detected_answers"]
    token_label_ids = np.zeros((len(token_starts)))
    for ans in answers:
        # look for indexes of the tokens which contain start and end of matched text
        start_idx = np.searchsorted(token_starts, ans["start"] + 1, 'left') - 1
        end_idx = np.searchsorted(token_starts, ans["end"], 'left')
        token_label_ids[start_idx:end_idx] = 1

    return {'token_label_ids': token_label_ids}


class _NormBboxesInput(TypedDict):
    bboxes: List[List[int]]
    page_bboxes: List[List[int]]


T = TypeVar('T', bound=_NormBboxesInput)


@feed_single_example
def norm_bboxes_for_layoutlm(example: T, **kwargs) -> T:
    bboxes, page_bboxes, page_spans = example['bboxes'], example['page_bboxes'], example['page_spans']
    norm_bboxes, norm_page_bboxes = _norm_bboxes_for_layoutlm(bboxes, page_bboxes, page_spans)

    return {'bboxes': norm_bboxes,
            'page_bboxes': norm_page_bboxes}


def _norm_bboxes_for_layoutlm(bboxes: List[List[int]],
                              page_bboxes: List[List[int]],
                              page_spans: List[Tuple[int, int]]) -> Tuple[List[List[float]], List[List[float]]]:
    norm_bboxes = np.array(bboxes)
    norm_page_bboxes = np.array(page_bboxes)
    for (_, _, _, page_height), (page_start_i, page_end_i) in zip(page_bboxes, page_spans):
        norm_bboxes[page_start_i:page_end_i, [1, 3]] = norm_bboxes[page_start_i:page_end_i, [1, 3]] / (page_height/1000)

    norm_page_bboxes[:, 3] = 1000

    return norm_bboxes, norm_page_bboxes


@feed_single_example
def stack_pages(example, **kwargs):
    bboxes = np.array(example['bboxes'])
    if example['token_page_nums'][0] == example['token_page_nums'][-1]:
        return {'bboxes': bboxes}
    page_nums = np.array(example['token_page_nums'])
    special_tokens_mask = np.array(example['special_tokens_mask'])

    # stack pages one under each other and squeeze them so the dimension is not greater than 1000
    page_offset = page_nums - page_nums[0]
    y_coord = bboxes[np.logical_not(special_tokens_mask)][:, (1, 3)] + page_offset[:, None] * 1000
    y_coord_norm = y_coord / (page_offset[-1] + 1)
    bboxes[np.logical_not(special_tokens_mask), 1] = y_coord_norm[:, 0]
    bboxes[np.logical_not(special_tokens_mask), 3] = y_coord_norm[:, 1]

    return {'bboxes': bboxes}
