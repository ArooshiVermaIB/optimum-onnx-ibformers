import numpy as np

from ibformers.data.utils import (
    spread_with_mapping,
    recalculate_spans,
    feed_batch,
    spread_with_first_token,
)


@feed_batch
def tokenize(example_batch, tokenizer, max_length=510, padding=False, **kwargs):
    # Tokenize contexts and questions (as pairs of inputs)

    encodings = tokenizer(
        example_batch["words"],
        is_split_into_words=True,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_token_type_ids=False,
        padding=padding,
    )

    # add token offsets for a document
    # add also word<->token mapping which indicate word index for given token
    batch_word_starts = []
    batch_word_map = []
    for i in range(len(encodings.encodings)):
        word_map = np.array(encodings.encodings[i].word_ids)
        word_starts = word_map != np.pad(word_map[:-1], pad_width=(1, 0), constant_values=-1)
        batch_word_map.append(word_map)
        batch_word_starts.append(word_starts)

    encodings["word_map"] = batch_word_map
    encodings["word_starts"] = batch_word_starts

    if "prefix_words" in example_batch:
        prefix_encodings = tokenizer(
            example_batch["prefix_words"],
            is_split_into_words=True,
            return_offsets_mapping=False,
            add_special_tokens=False,
            return_token_type_ids=False,
            padding=padding,
        )

        encodings["prefix_input_ids"] = prefix_encodings["input_ids"]
        if "prefix_mqa_ids" in example_batch:
            batch_prefix_word_map = []
            for i in range(len(prefix_encodings.encodings)):
                pref_word_map = np.array(prefix_encodings.encodings[i].word_ids)
                batch_prefix_word_map.append(pref_word_map)
            encodings["prefix_mqa_ids"] = spread_with_mapping(example_batch["prefix_mqa_ids"], batch_prefix_word_map)

        if "question_positions" in example_batch:
            pref_word_map = [np.array(pref_enc.word_ids) for pref_enc in prefix_encodings.encodings]
            encodings["question_positions"] = recalculate_spans(example_batch["question_positions"], pref_word_map)

    # bboxes need to be spread
    if "bboxes" in example_batch:
        encodings["bboxes"] = spread_with_mapping(example_batch["bboxes"], encodings["word_map"])

    # page_nums need to be spread
    if "word_page_nums" in example_batch:
        encodings["token_page_nums"] = spread_with_mapping(example_batch["word_page_nums"], encodings["word_map"])

    # token labels as well - use only first token of the word as a label
    if "token_label_ids" in example_batch:
        encodings["token_label_ids"] = spread_with_first_token(
            example_batch["token_label_ids"], encodings["word_map"], encodings["word_starts"]
        )

    if "answer_token_label_ids" in example_batch:
        encodings["answer_token_label_ids"] = spread_with_first_token(
            example_batch["answer_token_label_ids"], encodings["word_map"], encodings["word_starts"]
        )

    # page spans need to be adjusted for new token ranges
    if "page_spans" in example_batch:
        new_token_spans_batch = recalculate_spans(example_batch["page_spans"], encodings["word_map"])
        encodings["page_spans"] = new_token_spans_batch

    if "entities" in example_batch:
        # get batch of spans
        entities_batch = example_batch["entities"]
        for i in range(len(entities_batch)):
            word_map = [encodings["word_map"][i]]
            tokens_span = entities_batch[i]["token_spans"]
            new_token_spans = recalculate_spans(tokens_span, word_map)
            entities_batch[i]["token_spans"] = new_token_spans
        encodings["entities"] = entities_batch

    if "word_record_idx" in example_batch:
        encodings["word_record_idx"] = spread_with_mapping(example_batch["word_record_idx"], encodings["word_map"])

    return encodings


@feed_batch
def tokenize_layoutlmv2(example_batch, tokenizer, padding=False, **kwargs):
    # Tokenize contexts and questions (as pairs of inputs)

    encodings = tokenizer(
        example_batch["words"],
        boxes=example_batch["bboxes"],
        word_labels=example_batch["token_label_ids"],
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_token_type_ids=False,
        padding=padding,
    )

    # add token offsets for a document
    # add also word<->token mapping which indicate word index for given token
    batch_word_starts = []
    batch_word_map = []
    for i in range(len(encodings.encodings)):
        word_map = np.array(encodings.encodings[i].word_ids)
        word_starts = word_map != np.roll(word_map, shift=1)
        batch_word_map.append(word_map)
        batch_word_starts.append(word_starts)

    encodings["word_map"] = batch_word_map
    encodings["word_starts"] = batch_word_starts

    # rename keys
    encodings["bboxes"] = [np.array(bboxes) for bboxes in encodings.pop("bbox")]
    encodings["token_label_ids"] = encodings.pop("labels")

    if "prefix_words" in example_batch:
        lens_prefix = [len(pwords) for pwords in example_batch["prefix_words"]]
        prefix_encodings = tokenizer(
            example_batch["prefix_words"],
            boxes=[[[0, 0, 0, 0]] * lpref for lpref in lens_prefix],
            word_labels=None,
            return_offsets_mapping=False,
            add_special_tokens=False,
            return_token_type_ids=False,
            padding=padding,
        )

        encodings["prefix_input_ids"] = prefix_encodings["input_ids"]
        if "prefix_mqa_ids" in example_batch:
            batch_prefix_word_map = []
            for i in range(len(prefix_encodings.encodings)):
                pref_word_map = np.array(prefix_encodings.encodings[i].word_ids)
                batch_prefix_word_map.append(pref_word_map)
            encodings["prefix_mqa_ids"] = spread_with_mapping(example_batch["prefix_input_ids"], batch_prefix_word_map)

    # page_nums need to be spread
    if "word_page_nums" in example_batch:
        encodings["token_page_nums"] = spread_with_mapping(example_batch["word_page_nums"], encodings["word_map"])

    # page spans need to be adjusted for new token ranges
    if "page_spans" in example_batch:
        new_token_spans_batch = recalculate_spans(example_batch["page_spans"], encodings["word_map"])
        encodings["page_spans"] = new_token_spans_batch

    if "entities" in example_batch:
        # get batch of spans
        entities_batch = example_batch["entities"]
        for i in range(len(entities_batch)):
            word_map = [encodings["word_map"][i]]
            tokens_span = entities_batch[i]["token_spans"]
            new_token_spans = recalculate_spans(tokens_span, word_map)
            entities_batch[i]["token_spans"] = new_token_spans
        encodings["entities"] = entities_batch

    return encodings
