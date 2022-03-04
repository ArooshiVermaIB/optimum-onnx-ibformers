import logging
from random import shuffle
from typing import List, TypeVar, Tuple
import numpy as np
from typing_extensions import TypedDict

from ibformers.data.utils import (
    convert_to_dict_of_lists,
    tag_answer_in_doc,
    feed_single_example,
    feed_single_example_and_flatten,
    get_tokens_spans,
    convert_to_list_of_dicts,
)


@feed_single_example
def fuzzy_tag_in_document(example, **kwargs):
    # try to find an answer inside the text of the document
    # example will be skipped in case of no spans found
    words, entities = example["words"], example["entities"]
    if isinstance(entities, dict):
        lst_entities = convert_to_list_of_dicts(entities)
    else:
        lst_entities = entities
    words_len = list(map(len, words))
    word_offsets = np.cumsum(np.array([-1] + words_len[:-1]) + 1)
    # iterate over multiple questions
    new_entities = []
    dummy_tok_lab_id = 1
    for ent_id, ent in enumerate(lst_entities):
        question, answer = ent["name"], ent["text"]
        if len(ent["token_spans"]) == 0:
            detected_answer = tag_answer_in_doc(words=words, answer=answer)
            if len(detected_answer) == 0:
                continue
            token_spans = get_tokens_spans([[m["start"], m["end"]] for m in detected_answer], word_offsets)
        else:
            # leave original entity if token_spans are already found
            token_spans = ent["token_spans"]
        new_ent = ent
        new_ent["token_spans"] = token_spans
        new_ent["token_label_id"] = dummy_tok_lab_id
        dummy_tok_lab_id += 1
        new_entities.append(new_ent)

    if len(new_entities) == 0:
        return None
    new_dict_entities = convert_to_dict_of_lists(new_entities, list(ent.keys()))
    return {"entities": new_dict_entities}


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


@feed_single_example
def build_prefix_with_mqa_ids(example, tokenizer, shuffle_mqa_ids=False, convert_to_question=True, **kwargs):
    entities = example["entities"]
    mqa_size = 20
    pad_mqa_id = 1

    # limit entities to max 17 ent
    # entities = {k: v[:17] for k, v in entities.items()}

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
    if len(prefix) > mqa_size - 1:
        raise ValueError(f"There are {len(prefix)} entities detected. Thats too much for MQA model")
    # if shuffle_mqa_ids:
    #     shuffle(prefix)
    # make it sound like a natural question
    if convert_to_question:
        prefix = [f"what is the {ent.replace('_', ' ')}?" for ent in prefix]
    prefix = prefix + [tokenizer.sep_token]
    mqa_ids = used_mqa_ids + [1]

    # check if for each entity we chose unique token
    assert len(used_mqa_ids) == len(set(used_mqa_ids)), "mqa_id was re-used for more than one entity class"

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


@feed_single_example_and_flatten
def build_prefix_single_qa(example, tokenizer, **kwargs):
    """
    Create an example per question as required by QA model, and also keep
    one entity per example for simplicity and to avoid any confusion later.
    """
    entities = example["entities"]
    for q_idx, question in enumerate(entities["name"]):
        new_example = {
            "prefix_words": question.split() + [tokenizer.sep_token],
            # Get features of a particular question only
            "entities": {k: [v[q_idx]] for k, v in entities.items()},
        }
        yield {**example, **new_example}


@feed_single_example
def token_spans_to_start_end(example, **kwargs):
    """
    Create start and end positions of answer from entity token span. Token
    spans are relative to context tokens only but start and end positions in QA
    should be relative to model input (question/prefix + context) tokens.
    """
    # TODO: possibly refactor this fuction as this would require entities saved for each chunk
    #  while, with save_memory=True entities are saved only for first chunk in doc
    token_spans = example["entities"]["token_spans"][0]
    if not token_spans:  # no answer
        return {"start_positions": 0, "end_positions": 0}

    tok_start, tok_end = token_spans[0]
    chunk_start, chunk_end = example["chunk_ranges"]
    context_start = np.where(example["content_tokens_mask"])[0][0]

    # Check if the answer is in this chunk complelely
    if chunk_start <= tok_start and tok_end <= chunk_end:
        tok_start += context_start - chunk_start
        tok_end += context_start - chunk_start - 1  # inclusive of end word token
    else:
        tok_start, tok_end = 0, 0

    return {"start_positions": tok_start, "end_positions": tok_end}


@feed_single_example_and_flatten
def prepare_input_squad(example, tokenizer, **kwargs):
    """
    Create an example per question as required by QA model, and also keep
    one entity per example for simplicity and to avoid any confusion later.
    """
    entities = example["entities"]
    for q_idx, question in enumerate(entities["name"]):
        new_example = {
            "prefix_words": question.split() + [tokenizer.sep_token],
            # Get features of a particular question only
            "entities": {k: [v[q_idx]] for k, v in entities.items()},
        }
        if len(entities["token_spans"]) != 1:
            continue
        yield {**example, **new_example}


@feed_single_example
def convert_from_mrqa_fmt(example, **kwargs):
    if "entities" in example:
        return {}

    words = example["context_tokens"]["tokens"]
    question = example["question"]

    spans = list(
        set([(st, en + 1) for s in example["detected_answers"]["token_spans"] for st, en in zip(s["start"], s["end"])])
    )
    answers = example["answers"]
    answers_low = [a.lower().replace(" ", "") for a in answers]

    occupied_idx = np.zeros((len(words)))
    dummy_tok_lab_id = 1
    entities = []

    for s in spans:
        text = " ".join(words[s[0] : s[1]])
        if text.strip().lower().replace(" ", "") not in answers_low:
            logging.error(f"did not found {text.strip()} in possible answers {answers}")
        # don't include spans which were already included at some part before
        if occupied_idx[s[0] : s[1]].sum() > 0:
            continue
        occupied_idx[s[0] : s[1]] = 1
        lab = {
            "name": question,
            "order_id": 0,
            "text": text,
            "token_spans": [s],
            "token_label_id": dummy_tok_lab_id,
        }  # add dummy label id
        dummy_tok_lab_id += 1
        entities.append(lab)

    return {"words": words, "entities": entities}
