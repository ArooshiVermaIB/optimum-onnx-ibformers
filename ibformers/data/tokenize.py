import numpy as np

from ibformers.data.utils import spread_with_mapping, recalculate_spans


def tokenize(example_batch, tokenizer, **kwargs):
    # Tokenize contexts and questions (as pairs of inputs)
    encodings = tokenizer(example_batch['words'],
                          is_split_into_words=True,
                          return_offsets_mapping=True,
                          add_special_tokens=False,
                          return_token_type_ids=False,
                          # TODO: remove this once splitter processors are developed
                          max_length=512,
                          padding=False,
                          )

    # add token offsets for a document
    # add also word<->token mapping which indicate word index for given token
    batch_token_starts = []
    batch_word_map = []
    for single_offset in encodings['offset_mapping']:
        offset_mapping = np.array(single_offset)
        token_lens = offset_mapping[:, 1] - offset_mapping[:, 0]
        # space positions are just before new word which is indicating by 0 occuring in the first position
        new_word_positions = (offset_mapping[:, 0] == 0).astype(np.int)
        word_map = np.cumsum(new_word_positions) - 1
        # space positions are just before a new word
        space_positions = np.roll(new_word_positions, shift=-1)
        # compute token start position in the document, add 0 at the begining (first token starts at 0)
        token_starts_in_doc = np.pad(np.cumsum(token_lens + space_positions)[:-1], (1, 0))

        batch_token_starts.append(token_starts_in_doc)
        batch_word_map.append(word_map)

    encodings["token_character_starts"] = batch_token_starts
    encodings["word_map"] = batch_word_map

    # bboxes need to be spreaded
    if 'bboxes' in example_batch:
        encodings['bboxes'] = spread_with_mapping(example_batch['bboxes'], encodings['word_map'])

    # token labels as well
    if 'token_label_ids' in example_batch:
        encodings['token_label_ids'] = spread_with_mapping(example_batch['token_label_ids'], encodings['word_map'])

    # page spans need to be adjusted for new token ranges
    if 'page_spans' in example_batch:
        new_token_spans_batch = recalculate_spans(example_batch['page_spans'], encodings['word_map'])
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