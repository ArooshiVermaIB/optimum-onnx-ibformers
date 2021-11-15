import logging
from typing import TypeVar, List, Sequence, Any, Mapping, Tuple

import numpy as np

from ibformers.data.utils import (
    feed_single_example,
    convert_to_dict_of_lists,
    feed_single_example_and_flatten,
)


def get_chunk_ranges(input_len, chunk_size, overlap):
    if input_len < chunk_size:
        return [(0, input_len)]
    ranges = []
    for i in range(0, input_len - 1, chunk_size):
        from_range = max(0, i - overlap)
        to_range = min(input_len, i + chunk_size)
        ranges.append((from_range, to_range))

    return ranges


def get_single_page_chunk_ranges(input_len, chunk_size, overlap, page_nums):
    assert input_len == len(page_nums)

    page_nums_arr = np.array(page_nums)
    ranges = np.nonzero(np.diff(page_nums_arr, prepend=-2, append=-1))[0]

    all_ranges = []
    for i in range(len(ranges) - 1):
        rng_from, rng_to = ranges[i], ranges[i + 1]
        page_len = rng_to - rng_from
        page_chunks = get_chunk_ranges(page_len, chunk_size, overlap)
        for page_chunk in page_chunks:
            all_ranges.append((page_chunk[0] + rng_from, page_chunk[1] + rng_from))

    return all_ranges


@feed_single_example_and_flatten
def produce_chunks(
    example, tokenizer, max_length, chunking_strategy="ALL_CHUNKS", chunk_overlap=64, **kwargs
) -> Sequence:
    prefix_len = len(example.get("prefix_input_ids", []))
    input_len = len(example.get("input_ids", []))

    assert max_length > 0
    assert chunk_overlap >= 0
    assert chunk_overlap < max_length
    if chunk_overlap + prefix_len < (max_length // 2):
        logging.warning(
            f"Extra tokens occupies too much space. Prefix tokens: {prefix_len}, overlap: {chunk_overlap}"
        )

    if chunking_strategy == "ALL_CHUNKS":
        chunk_ranges = get_chunk_ranges(input_len, max_length - prefix_len - 2, chunk_overlap)
        return get_chunks(example, tokenizer, chunk_ranges)
    elif chunking_strategy == "SINGLE_PAGES":
        page_nums = example.get("token_page_nums")
        chunk_ranges = get_single_page_chunk_ranges(
            input_len, max_length - prefix_len - 2, chunk_overlap, page_nums
        )
        return get_chunks(example, tokenizer, chunk_ranges)
    else:
        raise ValueError(f"{chunking_strategy} is not implemented")


def get_chunks(example, tokenizer, chunk_ranges) -> Sequence[Mapping]:
    """
    Input ID: [1,2,3,4,5,6]
    Chunked: [1,2,3], [4,5,6]
    Chunk ranges: [0,3], [3,6] # Ranges in the original list
    # [0,1,2], [3,4,5]
    range(3,6) -> [3,4,5]
    """

    keys_to_chunk = [
        "input_ids",
        "bboxes",
        "token_label_ids",
        "offset_mapping",
        "word_starts",
        "word_map",
        "token_page_nums",
    ]

    # TODO: check how many tokens are added and remove hardcoded "2"
    chunked = {
        k: _split_by_ranges(example[k], ranges=chunk_ranges) for k in keys_to_chunk if k in example
    }

    # add images to chunks
    # get pages
    if 'images' in example:
        pages_idx = [sorted(set(pg)) for pg in chunked['token_page_nums']]
        assert all(
            [len(a) == 1 for a in pages_idx]
        ), "Chunks need to be single page to support images"
        pages_idx = np.array([a[0] for a in pages_idx])
        chunked['images'] = example['images'][pages_idx][:, None]

    chunked["chunk_ranges"] = chunk_ranges

    # This includes things like the document's ID
    other_keys = [
        i for i in list(example.keys()) if i not in set(keys_to_chunk).union(chunked.keys())
    ]

    # We're transposing now to make it easier to "flatten" the document into essentially independent examples
    transposed = [{k: v[i] for k, v in chunked.items()} for i, _ in enumerate(chunked["input_ids"])]

    # TODO: we might want to chunk also global objects like words, images etc.
    #  to not duplicate these large objects for each chunk
    transposed_plus_other_keys = [{**i, **{k: example[k] for k in other_keys}} for i in transposed]

    for chunk in transposed_plus_other_keys:
        # For some reason, return_special_tokens_mask=True doesn't work correctly here...
        # doing it in two steps is a workaround
        chunk_input_ids = example.get("prefix_input_ids", []) + chunk["input_ids"]
        chunk_processed = tokenizer.prepare_for_model(chunk_input_ids, add_special_tokens=True)
        chunk_processed = {**chunk, **chunk_processed}

        special_mask = np.array(
            tokenizer.get_special_tokens_mask(
                chunk_processed["input_ids"], already_has_special_tokens=True
            )
        )
        # do not treat UNK token as special
        unk_id = getattr(tokenizer, "unk_token_id", -1)
        special_mask = special_mask * (np.array(chunk_processed["input_ids"]) != unk_id)

        # all prefix tokens will be treated as special
        if "prefix_input_ids" in example:
            # search for prefix start
            prefix_tokens = example["prefix_input_ids"]
            chunk_tokens = chunk_processed["input_ids"]
            prefix_start = chunk_tokens.index(prefix_tokens[0])
            len_prefix = len(prefix_tokens)
            special_mask[prefix_start : prefix_start + len_prefix] = 1

            if "prefix_mqa_ids" in example:
                # fill values with padding idx
                mqa_ids = np.full_like(chunk_tokens, fill_value=1)
                mqa_ids[prefix_start : prefix_start + len_prefix] = example["prefix_mqa_ids"]
                chunk_processed["mqa_ids"] = mqa_ids

        content_tokens_mask = np.logical_not(special_mask)

        if len(chunk["input_ids"]) != content_tokens_mask.sum():
            raise ValueError(
                f'Number of non special tokens should be equal to number of chunk tokens. '
                f'chunk_input={chunk["input_ids"]}, special_mask={special_mask}'
            )

        chunk_processed["content_tokens_mask"] = content_tokens_mask
        chunk_processed["bboxes"] = fill_special_tokens(
            chunk["bboxes"], content_tokens_mask, 0
        ).tolist()
        # if "prefix_input_ids" in example:
        #     prefix_bboxes = np.array(
        #         [[[i * 20, 10, i * 20 + 10, 20] for i in range(1, len_prefix + 1)]]
        #     )
        #     chunk_processed["bboxes"][prefix_start : prefix_start + len_prefix] = prefix_bboxes

        chunk_processed["token_label_ids"] = fill_special_tokens(
            chunk["token_label_ids"], content_tokens_mask, -100
        )

        yield chunk_processed


# def first_only(example, tokenizer, max_length: int):
#     chunks = tokenizer.prepare_for_model(example["input_ids"], max_length=max_length,
#                                          add_special_tokens=True)
#     special_mask = np.array(tokenizer.get_special_tokens_mask(chunks["input_ids"], already_has_special_tokens=True))
#     chunks['special_tokens_mask'] = special_mask
#
#     max_len_wo_special = len(special_mask) - special_mask.sum()
#     chunks['offset_mapping'] = example['offset_mapping'][:max_len_wo_special]
#     chunks['word_map'] = example['word_map'][:max_len_wo_special]
#     chunks['word_starts'] = example['word_starts'][:max_len_wo_special]
#     chunks['token_page_nums'] = example['token_page_nums'][:max_len_wo_special]
#
#     if 'bboxes' in example:
#         chunks["bboxes"] = np.array(example["bboxes"])[:max_len_wo_special]
#         chunks["bboxes"] = fill_special_tokens(chunks["bboxes"], special_mask, 0)
#
#     if 'token_label_ids' in example:
#         chunks["token_label_ids"] = np.array(example["token_label_ids"])[:max_len_wo_special]
#         chunks['token_label_ids'] = fill_special_tokens(chunks["token_label_ids"], special_mask, -100)
#
#     return [chunks]

# return convert_to_dict_of_lists([chunks, chunks], keys=list(chunks.keys()))


# example = {"input_ids": [1, 2, 3, 4],
#            "offset_mapping": [(0, 1), (2, 3), (4, 5), (6, 7)],
#            "word_map": [0, 0, 1, 1],
#            "bboxes": [[0, 0, 1, 1]] * 4,
#            "token_label_ids": [1, 2, 3, 4]}
# produce_chunks(example, )


def fill_special_tokens(arr: Sequence[Any], content_mask: Sequence[int], fill_value: int):
    arr = np.array(arr)
    new_dims = [len(content_mask)] + list(arr.shape[1:])
    filled = np.full_like(arr, shape=new_dims, fill_value=fill_value)
    filled[content_mask] = arr

    return filled


def _split_by_ranges(seq: Sequence[Any], ranges: List[Tuple]):
    if len(seq) != ranges[-1][-1]:
        raise ValueError(f"Seqence of length {len(seq)} does not match ranges {ranges}")
    chunks = []
    for rng_from, rng_to in ranges:
        chunks.append(seq[rng_from:rng_to])

    return chunks
