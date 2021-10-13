from typing import List, TypeVar, Tuple

import numpy as np
from typing_extensions import TypedDict

from ibformers.data.utils import convert_to_dict_of_lists, tag_answer_in_doc, feed_single_example
from random import shuffle


@feed_single_example
def fuzzy_tag_in_document(example, **kwargs):
    # try to find an answer inside the text of the document
    # example will be skipped in case of no spans found
    answers = tag_answer_in_doc(words=example["words"], answer=example["answer"])
    if len(answers) == 0:
        return None
    else:
        return {"detected_answers": answers}


@feed_single_example
def add_token_labels_qa(example, **kwargs):
    token_starts = example["token_starts"]
    answers = example["detected_answers"]
    token_label_ids = np.zeros((len(token_starts)))
    for ans in answers:
        # look for indexes of the tokens which contain start and end of matched text
        start_idx = np.searchsorted(token_starts, ans["start"] + 1, "left") - 1
        end_idx = np.searchsorted(token_starts, ans["end"], "left")
        token_label_ids[start_idx:end_idx] = 1

    return {"token_label_ids": token_label_ids}


class _NormBboxesInput(TypedDict):
    bboxes: List[List[int]]
    page_bboxes: List[List[int]]


T = TypeVar("T", bound=_NormBboxesInput)


@feed_single_example
def norm_bboxes_for_layoutlm(example: T, **kwargs) -> T:
    bboxes, page_bboxes, page_spans = (
        example["bboxes"],
        example["page_bboxes"],
        example["page_spans"],
    )
    norm_bboxes, norm_page_bboxes = _norm_bboxes_for_layoutlm(bboxes, page_bboxes, page_spans)

    return {"bboxes": norm_bboxes, "page_bboxes": norm_page_bboxes}


def _norm_bboxes_for_layoutlm(
    bboxes: List[List[int]], page_bboxes: List[List[int]], page_spans: List[Tuple[int, int]]
) -> Tuple[List[List[float]], List[List[float]]]:
    norm_bboxes = np.array(bboxes)
    norm_page_bboxes = np.array(page_bboxes)
    for (_, _, _, page_height), (page_start_i, page_end_i) in zip(page_bboxes, page_spans):
        norm_bboxes[page_start_i:page_end_i, [1, 3]] = norm_bboxes[
            page_start_i:page_end_i, [1, 3]
        ] / (page_height / 1000)

    norm_page_bboxes[:, 3] = 1000

    return norm_bboxes, norm_page_bboxes


@feed_single_example
def stack_pages(example, **kwargs):
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
def build_prefix_with_special_tokens(example, tokenizer, shuffle_extra_tokens=True, **kwargs):
    entities = example["entities"]
    all_extra_tokens = tokenizer.additional_special_tokens
    special_token_to_extra_id = {tok: idx for idx, tok in enumerate(all_extra_tokens)}

    # <extra_0> token will be reserved for O class
    available_extra_tokens = all_extra_tokens[1:10]

    if shuffle_extra_tokens:
        shuffle(available_extra_tokens)

    used_extra_tokens = []
    used_special_ids = []
    # get mapping of extra token to each entity
    for ent_id in entities["token_label_id"]:
        assert ent_id != 0, "Something wrong. 0 should be reserved for O class"
        extra_token = available_extra_tokens[ent_id]
        used_extra_tokens.append(extra_token)
        used_special_ids.append(special_token_to_extra_id[extra_token])

    prefix = [f"{tok} {name} {tok}" for name, tok in zip(entities["name"], used_extra_tokens)]
    if shuffle_extra_tokens:
        shuffle(prefix)
    prefix = prefix + [tokenizer.sep_token]

    # check if for each entity we chose unique token
    assert len(used_extra_tokens) == len(
        set(used_extra_tokens)
    ), "An extra token was re-used for more than one entity class"

    entities["extra_tokens"] = used_extra_tokens
    entities["used_label_id"] = used_special_ids

    # build token_label_ids
    token_label_ids = np.zeros((len(example["words"])), dtype=np.int64)
    for spans, tok in zip(entities["token_spans"], used_extra_tokens):
        extra_id = special_token_to_extra_id[tok]
        for span in spans:
            token_label_ids[span[0] : span[1]] = extra_id

    return {"entities": entities, "token_label_ids": token_label_ids, "prefix_words": prefix}


@feed_single_example
def build_prefix_with_mqa_ids(example, tokenizer, shuffle_mqa_ids=True, **kwargs):
    entities = example["entities"]
    mqa_size = 10
    pad_mqa_id = 1

    # all_extra_tokens = tokenizer.additional_special_tokens
    # special_token_to_extra_id = {tok: idx for idx, tok in enumerate(all_extra_tokens)}

    # 0 idx is reserved for O class, 1 idx is reserved for padding
    available_mqa_ids = list(range(2, mqa_size))

    if shuffle_mqa_ids:
        shuffle(available_mqa_ids)

    used_mqa_ids = []
    # get mapping of extra token to each entity
    for ent_id in entities["token_label_id"]:
        assert ent_id != 0, "Something wrong. 0 should be reserved for O class"
        mqa_id = available_mqa_ids[ent_id]
        used_mqa_ids.append(mqa_id)

    prefix = entities["name"]
    # if shuffle_mqa_ids:
    #     shuffle(prefix)
    prefix = prefix + [tokenizer.sep_token]
    mqa_ids = used_mqa_ids + [1]

    # check if for each entity we chose unique token
    assert len(used_mqa_ids) == len(
        set(used_mqa_ids)
    ), "mqa_id was re-used for more than one entity class"

    entities["used_label_id"] = used_mqa_ids

    # build token_label_ids
    token_label_ids = np.zeros((len(example["words"])), dtype=np.int64)
    for spans, mqa_id in zip(entities["token_spans"], used_mqa_ids):
        for span in spans:
            token_label_ids[span[0] : span[1]] = mqa_id

    return {
        "entities": entities,
        "token_label_ids": token_label_ids,
        "prefix_words": prefix,
        "prefix_mqa_ids": mqa_ids,
    }


@feed_single_example
def build_prefix_with_mqa_ids(example, tokenizer, shuffle_mqa_ids=True, **kwargs):
    entities = example["entities"]
    mqa_size = 10
    pad_mqa_id = 1

    # all_extra_tokens = tokenizer.additional_special_tokens
    # special_token_to_extra_id = {tok: idx for idx, tok in enumerate(all_extra_tokens)}

    # 0 idx is reserved for O class, 1 idx is reserved for padding
    available_mqa_ids = list(range(2, mqa_size))

    if shuffle_mqa_ids:
        shuffle(available_mqa_ids)

    used_mqa_ids = []
    # get mapping of extra token to each entity
    for ent_id in entities["token_label_id"]:
        assert ent_id != 0, "Something wrong. 0 should be reserved for O class"
        mqa_id = available_mqa_ids[ent_id]
        used_mqa_ids.append(mqa_id)

    prefix = entities["name"]
    # if shuffle_mqa_ids:
    #     shuffle(prefix)
    prefix = prefix + [tokenizer.sep_token]
    mqa_ids = used_mqa_ids + [1]

    # check if for each entity we chose unique token
    assert len(used_mqa_ids) == len(
        set(used_mqa_ids)
    ), "mqa_id was re-used for more than one entity class"

    entities["used_label_id"] = used_mqa_ids

    # build token_label_ids
    token_label_ids = np.zeros((len(example["words"])), dtype=np.int64)
    for spans, mqa_id in zip(entities["token_spans"], used_mqa_ids):
        for span in spans:
            token_label_ids[span[0] : span[1]] = mqa_id

    return {
        "entities": entities,
        "token_label_ids": token_label_ids,
        "prefix_words": prefix,
        "prefix_mqa_ids": mqa_ids,
    }
