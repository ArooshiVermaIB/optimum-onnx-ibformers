from typing import List, Tuple, TypeVar

import numpy as np
from fuzzysearch import find_near_matches
from typing_extensions import TypedDict


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


class _NormBboxesInput(TypedDict):
    bboxes: List[List[List[float]]]
    page_bboxes: List[List[List[float]]]


T = TypeVar('T', bound=_NormBboxesInput)


def norm_bboxes_for_layoutlm(example_batch: T) -> T:
    bboxes_batch = []
    page_bboxes_batch = []
    for bboxes, page_bboxes, page_spans in zip(example_batch['bboxes'],
                                               example_batch['page_bboxes'],
                                               example_batch['page_spans']):
        norm_bboxes, norm_page_bboxes = _norm_bboxes_for_layoutlm(bboxes, page_bboxes, page_spans)
        bboxes_batch.append(norm_bboxes)
        page_bboxes_batch.append(norm_page_bboxes)

    return {'bboxes': bboxes_batch,
            'page_bboxes': page_bboxes_batch}


def _norm_bboxes_for_layoutlm(bboxes: List[List[float]],
                              page_bboxes: List[List[float]],
                              page_spans: List[Tuple[int, int]]) -> Tuple[List[List[float]], List[List[float]]]:
    norm_bboxes = np.array(bboxes)
    norm_page_bboxes = np.array(page_bboxes)
    for (_, _, _, page_height), (page_start_i, page_end_i) in zip(page_bboxes, page_spans):
        norm_bboxes[page_start_i:page_end_i, [1, 3]] = norm_bboxes[page_start_i:page_end_i, [1, 3]] / page_height

    norm_bboxes = (norm_bboxes * 1000).round().astype(np.int)
    norm_page_bboxes[:, 3] = 1
    norm_page_bboxes *= 1000

    return norm_bboxes, norm_page_bboxes
