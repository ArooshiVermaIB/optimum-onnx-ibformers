import numpy as np
from fuzzysearch import find_near_matches


def feed_single_example(fn):
    """
    Examples processed by map method of hf/datasets are processed in batches.
    This is a helper function/decorator to use if you want to get single examples instead of
    whole batch
    :param fn: function to decorate
    :return: batch of examples updated with function results
    """
    def split_batch(batch, **kwargs):
        batch_keys = list(batch.keys())
        len_of_batch = len(batch[batch_keys[0]])
        outs = []
        for i in range(len_of_batch):
            item_dict = {k: v[i] for k, v in batch.items()}
            out = fn(item_dict, **kwargs)
            if out is None:
                continue
            outs.append(out)
        out_keys = [] if len(out) == 0 else list(outs[0].keys())
        dict_of_lists = convert_to_dict_of_lists(outs, out_keys)
        batch.update(dict_of_lists)
        return batch
    return split_batch


def feed_batch(fn):
    """
    Function which is wrapped by this decorator might return only part of keys delivered in the input
    This function is making sure that we return both modified/added data and unchanged data from the input
    :param fn: function to decorate
    :return: batch of examples updated with function results
    """
    def update_batch(batch, **kwargs):
        out = fn(batch, **kwargs)
        batch.update(out)
        return batch
    return update_batch


def convert_to_dict_of_lists(list_of_dicts, keys):
    v = {k: [dic[k] for dic in list_of_dicts] for k in keys}
    return v


def get_tokens_spans(char_spans, token_offsets):
    """
    Function takes an input with character level spans and transform it to token spans.
    Example:

    :param char_spans: List[List[int]] character spans as a List of number pairs
    :param token_offsets: List[int]
    :return:
    """
    # indentify token indices in matches
    token_spans = []
    for span in char_spans:
        # look for indexes of the words which contain start and end of matched text
        start_idx = np.searchsorted(token_offsets, span[0] + 1, 'left') - 1
        end_idx = np.searchsorted(token_offsets, span[1], 'left')
        token_spans.append((start_idx, end_idx))

    return token_spans


def find_matches_in_text(text, answer, only_best=True):
    max_distance = int(len(answer) / 10)
    matches = find_near_matches(answer.lower().strip(), text.lower(), max_l_dist=max_distance)

    if len(matches) == 0:
        return []
    elif only_best:
        # only keep best matches
        best_match_distance = min([match.dist for match in matches])
        selected = list(filter(lambda x: x.dist == best_match_distance, matches))
    else:
        selected = matches

    # convert to list of dicts
    # correct text with original casing
    matches_dict = [{'text': text[m.start:m.end], 'start': m.start, 'end': m.end} for m in selected]

    return matches_dict


def tag_answer_in_doc(words, answer):
    # for very short answers finding correct span in the document might be difficult - it may results with matches
    # which are incorrect, better to skip such examples
    if len(answer.strip()) < 3:
        return []
    words_len = list(map(len, words))
    # compute offsets, add 1 to include space delimiter
    word_offsets = np.cumsum(np.array([-1] + words_len[:-1]) + 1)
    text = ' '.join(words)
    matches = find_matches_in_text(text, answer)
    # TODO: maybe add word spans, if it will be useful
    # token_spans = get_tokens_spans(matches, word_offsets)

    return matches


def spread_with_mapping(features_batch, word_map_batch):
    spread_features_batch = []
    for features, word_map in zip(features_batch, word_map_batch):
        features = np.array(features)
        spread_features = np.take(features, word_map, axis=0)
        spread_features_batch.append(spread_features)

    return spread_features_batch


def recalculate_spans(orig_spans_batch, word_map_batch):
    assert len(orig_spans_batch) == len(word_map_batch) or len(word_map_batch) == 1, \
        "Word map length should be either equal to spans, or global for all spans"
    recalculated_spans_batch = []
    for span_idx, span in enumerate(orig_spans_batch):
        span = np.array(span)
        word_map = word_map_batch[0] if len(word_map_batch) == 1 else word_map_batch[span_idx]
        recalculated_span = np.searchsorted(word_map, span, 'left')
        recalculated_spans_batch.append(recalculated_span)

    return recalculated_spans_batch

