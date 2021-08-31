import numpy as np

from ibformers.data.utils import feed_single_example


@feed_single_example
def produce_chunks(example, tokenizer, max_length, chunking_strategy="FIRST_ONLY", **kwargs):

    assert chunking_strategy == "FIRST_ONLY", "Only FIRST_ONLY chunking strategy is supported"
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

    return chunks


def fill_special_tokens(arr, special_mask, fill_value):
    new_dims = [len(special_mask)] + list(arr.shape[1:])
    filled = np.full_like(arr, shape=new_dims, fill_value=fill_value)
    filled[np.logical_not(special_mask)] = arr

    return filled
