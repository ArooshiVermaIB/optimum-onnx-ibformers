from typing import TypeVar, List

import numpy as np

from ibformers.data.utils import feed_single_example, convert_to_dict_of_lists, feed_single_example_and_flatten


@feed_single_example_and_flatten
def produce_chunks(example, tokenizer, max_length, chunking_strategy="FIRST_ONLY", chunk_overlap=64, **kwargs):
    if chunking_strategy == "FIRST_ONLY":
        return first_only(example, tokenizer, max_length)
    elif chunking_strategy == "ALL_CHUNKS":
        return all_chunks(example, tokenizer, max_length, chunk_overlap)
    else:
        raise ValueError("Shit went down")


def all_chunks(example, tokenizer, max_length: int, overlap: int):
    """
    Input ID: [1,2,3,4,5,6]
    Chunked: [1,2,3], [4,5,6]
    Chunk ranges: [0,3], [3,6] # Ranges in the original list
    # [0,1,2], [3,4,5]
    range(3,6) -> [3,4,5]
    """
    # TODO: check how many tokens are added and reomve hardcodded "2"
    keys_to_chunk = ["input_ids", "bboxes", "token_label_ids", "offset_mapping", "token_character_starts", "word_map"]
    chunked = {k: _chunk_with_overlap(example[k],
                                      chunk_size=max_length - 2,
                                      overlap=overlap) for k in keys_to_chunk if k in example}
    other_keys = [i for i in list(example.keys()) if i not in keys_to_chunk]
    transposed = [{k: v[i] for k, v in chunked.items()} for i, _ in enumerate(chunked['input_ids'])]
    trasposed_plus_other_keys = [{**i, **{k: example[k] for k in other_keys}} for i in transposed]


    # chunks = tokenizer.prepare_for_model(example["input_ids"], max_length=max_length,
    #                                      add_special_tokens=True)
    # special_mask = np.array(tokenizer.get_special_tokens_mask(chunks["input_ids"], already_has_special_tokens=True))
    # chunks['special_tokens_mask'] = special_mask
    #
    # max_len_wo_special = len(special_mask) - special_mask.sum()
    # chunks['offset_mapping'] = example['offset_mapping'][:max_len_wo_special]
    # chunks['word_map'] = example['word_map'][:max_len_wo_special]
    #
    # if 'bboxes' in example:
    #     chunks["bboxes"] = np.array(example["bboxes"])[:max_len_wo_special]
    #     chunks["bboxes"] = fill_special_tokens(chunks["bboxes"], special_mask, 0)
    #
    # if 'token_label_ids' in example:
    #     chunks["token_label_ids"] = np.array(example["token_label_ids"])[:max_len_wo_special]
    #     chunks['token_label_ids'] = fill_special_tokens(chunks["token_label_ids"], special_mask, -100)

    return trasposed_plus_other_keys


def first_only(example, tokenizer, max_length: int):
    chunks = tokenizer.prepare_for_model(example["input_ids"], max_length=max_length,
                                         add_special_tokens=True)
    special_mask = np.array(tokenizer.get_special_tokens_mask(chunks["input_ids"], already_has_special_tokens=True))
    chunks['special_tokens_mask'] = special_mask

    max_len_wo_special = len(special_mask) - special_mask.sum()
    chunks['offset_mapping'] = example['offset_mapping'][:max_len_wo_special]
    chunks['word_map'] = example['word_map'][:max_len_wo_special]

    if 'bboxes' in example:
        chunks["bboxes"] = np.array(example["bboxes"])[:max_len_wo_special]
        chunks["bboxes"] = fill_special_tokens(chunks["bboxes"], special_mask, 0)

    if 'token_label_ids' in example:
        chunks["token_label_ids"] = np.array(example["token_label_ids"])[:max_len_wo_special]
        chunks['token_label_ids'] = fill_special_tokens(chunks["token_label_ids"], special_mask, -100)

    return [chunks]

    # return convert_to_dict_of_lists([chunks, chunks], keys=list(chunks.keys()))


# example = {"input_ids": [1, 2, 3, 4],
#            "offset_mapping": [(0, 1), (2, 3), (4, 5), (6, 7)],
#            "word_map": [0, 0, 1, 1],
#            "bboxes": [[0, 0, 1, 1]] * 4,
#            "token_label_ids": [1, 2, 3, 4]}
# produce_chunks(example, )


def fill_special_tokens(arr, special_mask, fill_value):
    new_dims = [len(special_mask)] + list(arr.shape[1:])
    filled = np.full_like(arr, shape=new_dims, fill_value=fill_value)
    filled[np.logical_not(special_mask)] = arr

    return filled


# Chunk with overlap
# Chunking without crossing page boundaries
# -> May lead to shit goin' down with things like relative biases

# [[1,2], [3,4]] -> [1,2,3,4]


S = TypeVar('S')


def _chunk_with_overlap(input_list: List[S], chunk_size: int, overlap: int) -> List[List[S]]:
    """
    Chunk a list, with a fixed amount of overlap (equal to 'stride')

    >>> _chunk_with_overlap(list(range(5)), chunk_size=10, overlap=0)
    [[0, 1, 2, 3, 4]]

    >>> _chunk_with_overlap(list(range(5)), chunk_size=2, overlap=0)
    [[0, 1], [2, 3], [4]]

    >>> _chunk_with_overlap(list(range(4)), chunk_size=2, overlap=0)
    [[0, 1], [2, 3]]

    >>> _chunk_with_overlap(list(range(10)), chunk_size=5, overlap=2)
    [[0, 1, 2, 3, 4], [3, 4, 5, 6, 7], [6, 7, 8, 9]]
    """
    assert chunk_size > 0
    assert overlap >= 0
    assert overlap < chunk_size
    if len(input_list) < chunk_size:
        return [input_list]
    l = []
    i = 0
    while i + overlap < len(input_list):
        x = input_list[i: i + chunk_size]
        l.append(x)
        i += chunk_size - overlap
    return l
