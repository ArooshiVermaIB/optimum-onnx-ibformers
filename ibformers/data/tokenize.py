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
        example_batch['words'],
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
        word_starts = word_map != np.roll(word_map, shift=1)
        batch_word_map.append(word_map)
        batch_word_starts.append(word_starts)

    encodings["word_map"] = batch_word_map
    encodings["word_starts"] = batch_word_starts

    # bboxes need to be spread
    if 'bboxes' in example_batch:
        encodings['bboxes'] = spread_with_mapping(example_batch['bboxes'], encodings['word_map'])

    # page_nums need to be spread
    if 'word_page_nums' in example_batch:
        encodings['token_page_nums'] = spread_with_mapping(
            example_batch['word_page_nums'], encodings['word_map']
        )

    # token labels as well - use only first token of the word as a label
    if 'token_label_ids' in example_batch:
        encodings['token_label_ids'] = spread_with_first_token(
            example_batch['token_label_ids'], encodings['word_starts']
        )

        # encodings['token_label_ids'] = spread_with_mapping(example_batch['token_label_ids'], encodings['word_map'])

    # page spans need to be adjusted for new token ranges
    if 'page_spans' in example_batch:
        new_token_spans_batch = recalculate_spans(
            example_batch['page_spans'], encodings['word_map']
        )
        encodings['page_spans'] = new_token_spans_batch

    if 'entities' in example_batch:
        # get batch of spans
        entities_batch = example_batch['entities']
        for i in range(len(entities_batch)):
            word_map = [encodings['word_map'][i]]
            tokens_span = entities_batch[i]['token_spans']
            new_token_spans = recalculate_spans(tokens_span, word_map)
            entities_batch[i]['token_spans'] = new_token_spans
        encodings['entities'] = entities_batch

    return encodings


@feed_batch
def tokenize_layoutlmv2(example_batch, tokenizer, padding=False, **kwargs):
    # Tokenize contexts and questions (as pairs of inputs)

    encodings = tokenizer(
        example_batch['words'],
        boxes=example_batch['bboxes'],
        word_labels=example_batch['token_label_ids'],
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
    encodings['bboxes'] = encodings.pop('bbox')
    encodings['token_label_ids'] = encodings.pop('labels')

    # page_nums need to be spread
    if 'word_page_nums' in example_batch:
        encodings['token_page_nums'] = spread_with_mapping(
            example_batch['word_page_nums'], encodings['word_map']
        )

    # page spans need to be adjusted for new token ranges
    if 'page_spans' in example_batch:
        new_token_spans_batch = recalculate_spans(
            example_batch['page_spans'], encodings['word_map']
        )
        encodings['page_spans'] = new_token_spans_batch

    if 'entities' in example_batch:
        # get batch of spans
        entities_batch = example_batch['entities']
        for i in range(len(entities_batch)):
            word_map = [encodings['word_map'][i]]
            tokens_span = entities_batch[i]['token_spans']
            new_token_spans = recalculate_spans(tokens_span, word_map)
            entities_batch[i]['token_spans'] = new_token_spans
        encodings['entities'] = entities_batch

    return encodings
