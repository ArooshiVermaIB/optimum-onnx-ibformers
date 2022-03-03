import logging
from collections import namedtuple, defaultdict
from typing import List

import numpy as np

from ibformers.data.chunk import fill_special_tokens, get_chunk_ranges
from ibformers.data.utils import (
    feed_single_example_and_flatten,
    feed_single_example,
    recalculate_spans,
    convert_to_list_of_dicts,
    convert_to_dict_of_lists,
)

# Code is mostly copied from Splinter repository
from ibformers.models.layv1splinter import SPLINTER_MAX_QUESTIONS

STOPWORDS = {
    "ourselves",
    "hers",
    "between",
    "yourself",
    "but",
    "again",
    "there",
    "about",
    "once",
    "during",
    "out",
    "very",
    "having",
    "with",
    "they",
    "own",
    "an",
    "be",
    "some",
    "for",
    "do",
    "its",
    "yours",
    "such",
    "into",
    "of",
    "most",
    "itself",
    "other",
    "off",
    "is",
    "s",
    "am",
    "or",
    "who",
    "as",
    "from",
    "him",
    "each",
    "the",
    "themselves",
    "until",
    "below",
    "are",
    "we",
    "these",
    "your",
    "his",
    "through",
    "don",
    "nor",
    "me",
    "were",
    "her",
    "more",
    "himself",
    "this",
    "down",
    "should",
    "our",
    "their",
    "while",
    "above",
    "both",
    "up",
    "to",
    "ours",
    "had",
    "she",
    "all",
    "no",
    "when",
    "at",
    "any",
    "before",
    "them",
    "same",
    "and",
    "been",
    "have",
    "in",
    "will",
    "on",
    "does",
    "yourselves",
    "then",
    "that",
    "because",
    "what",
    "over",
    "why",
    "so",
    "can",
    "did",
    "not",
    "now",
    "under",
    "he",
    "you",
    "herself",
    "has",
    "just",
    "where",
    "too",
    "only",
    "myself",
    "which",
    "those",
    "i",
    "after",
    "few",
    "whom",
    "t",
    "being",
    "if",
    "theirs",
    "my",
    "against",
    "a",
    "by",
    "doing",
    "it",
    "how",
    "further",
    "was",
    "here",
    "than",
    "also",
    "could",
    "would",
}
MaskedLmInstance = namedtuple("MaskedLmInstance", ["index", "label"])
MaskedSpanInstance = namedtuple("MaskedSpanInstance", ["index", "begin_label", "end_label"])


def _iterate_span_indices(span):
    return range(span[0], span[1] + 1)


def get_candidate_span_clusters(
    tokens: List[str], max_span_length: int, include_sub_clusters: bool = False, validate: bool = True
):
    """
    :param: tokens: List of chunk words passed for which recurring spans will be found
    :param max_span_length: maximum lenght measured in words of the detected span
    :param include_sub_clusters: whether to include not maximum size clusters
    :param validate: whether to filter out non interesting clusters
    :return:
    """
    token_to_indices = defaultdict(list)
    for i, token in enumerate(tokens):
        token_to_indices[token].append(i)

    recurring_spans = []
    for token, indices in token_to_indices.items():
        for i, idx1 in enumerate(indices):
            for j in range(i + 1, len(indices)):
                idx2 = indices[j]
                assert idx1 < idx2

                max_recurring_length = 1
                for length in range(1, max_span_length):
                    if include_sub_clusters:
                        recurring_spans.append((idx1, idx2, length))
                    if (idx2 + length) >= len(tokens) or tokens[idx1 + length] != tokens[idx2 + length]:
                        break
                    max_recurring_length += 1

                if max_recurring_length == max_span_length or not include_sub_clusters:
                    recurring_spans.append((idx1, idx2, max_recurring_length))

    spans_to_clusters = {}
    spans_to_representatives = {}
    for idx1, idx2, length in recurring_spans:
        first_span, second_span = (idx1, idx1 + length - 1), (idx2, idx2 + length - 1)
        if first_span in spans_to_representatives:
            if second_span not in spans_to_representatives:
                rep = spans_to_representatives[first_span]
                cluster = spans_to_clusters[rep]
                cluster.append(second_span)
                spans_to_representatives[second_span] = rep
        else:
            cluster = [first_span, second_span]
            spans_to_representatives[first_span] = first_span
            spans_to_representatives[second_span] = first_span
            spans_to_clusters[first_span] = cluster

    if validate:
        recurring_spans = [
            cluster
            for cluster in spans_to_clusters.values()
            if validate_ngram(tokens, cluster[0][0], cluster[0][1] - cluster[0][0] + 1)
        ]
    else:
        recurring_spans = spans_to_clusters.values()
    return recurring_spans


def validate_ngram(tokens: List[str], start_index: int, length: int):
    # We filter out n-grams that are all stopwords (e.g. "in the", "with my", ...)
    if any([tokens[idx].lower() not in STOPWORDS for idx in range(start_index, start_index + length)]):
        return True
    return False


def get_span_clusters_by_length(span_clusters, seq_length):
    already_taken = [False] * seq_length
    span_clusters = sorted(
        [(cluster, cluster[0][1] - cluster[0][0] + 1) for cluster in span_clusters], key=lambda x: x[1], reverse=True
    )
    filtered_span_clusters = []
    for span_cluster, _ in span_clusters:
        unpruned_spans = []
        for span in span_cluster:
            if any((already_taken[i] for i in range(span[0], span[1] + 1))):
                continue
            unpruned_spans.append(span)

        # Validating that the cluster is indeed "recurring" after the pruning
        if len(unpruned_spans) >= 2:
            filtered_span_clusters.append(unpruned_spans)
            for span in unpruned_spans:
                for idx in _iterate_span_indices(span):
                    already_taken[idx] = True

    return filtered_span_clusters


def create_recurring_span_selection_predictions(tokens, max_recurring_predictions, max_span_length, masked_lm_prob):
    masked_spans = []
    num_predictions = 0
    input_mask = [1] * len(tokens)
    new_tokens = list(tokens)

    already_masked_tokens = [False] * len(new_tokens)
    span_label_tokens = [False] * len(new_tokens)

    num_to_predict = min(max_recurring_predictions, max(1, int(round(len(tokens) * masked_lm_prob))))

    span_clusters = get_candidate_span_clusters(tokens, max_span_length, include_sub_clusters=True)
    span_clusters = get_span_clusters_by_length(span_clusters, len(tokens))
    span_clusters = [(cluster, tuple(tokens[cluster[0][0] : cluster[0][1] + 1])) for cluster in span_clusters]

    span_clusters = [span for span in span_clusters if is_interesting_span(span)]

    span_cluster_indices = np.random.permutation(range(len(span_clusters)))
    span_counter = 0
    while span_counter < len(span_cluster_indices):
        span_idx = span_cluster_indices[span_counter]
        span_cluster = span_clusters[span_idx][0]
        # self._assert_and_return_identical(token_ids, identical_spans)
        num_occurrences = len(span_cluster)

        unmasked_span_idx = np.random.randint(num_occurrences)
        unmasked_span = span_cluster[unmasked_span_idx]
        span_counter += 1
        if any([already_masked_tokens[i] for i in _iterate_span_indices(unmasked_span)]):
            # The same token can't be both masked for one pair and unmasked for another pair
            continue

        unmasked_span_beginning, unmasked_span_ending = unmasked_span
        for i, span in enumerate(span_cluster):
            if num_predictions >= num_to_predict:
                # logger.warning(f"Already masked {self.max_predictions} spans.")
                break

            if any([already_masked_tokens[j] for j in _iterate_span_indices(unmasked_span)]):
                break

            if i != unmasked_span_idx:
                if any([already_masked_tokens[j] or span_label_tokens[j] for j in _iterate_span_indices(span)]):
                    # The same token can't be both masked for one pair and unmasked for another pair,
                    # or alternatively masked twice
                    continue

                if any(
                    [
                        new_tokens[j] != new_tokens[k]
                        for j, k in zip(_iterate_span_indices(span), _iterate_span_indices(unmasked_span))
                    ]
                ):
                    logging.warning(
                        f"Two non-identical spans: unmasked {new_tokens[unmasked_span_beginning:unmasked_span_ending + 1]}, "
                        f"masked:{new_tokens[span[0]:span[1] + 1]}"
                    )
                    continue

                is_first_token = True
                for j in _iterate_span_indices(span):
                    if is_first_token:
                        new_tokens[j] = "[QUESTION]"
                        masked_spans.append(
                            MaskedSpanInstance(
                                index=span, begin_label=unmasked_span_beginning, end_label=unmasked_span_ending
                            )
                        )
                        num_predictions += 1
                    else:
                        new_tokens[j] = "[PAD]"
                        input_mask[j] = 0

                    is_first_token = False
                    already_masked_tokens[j] = True

                for j in _iterate_span_indices(unmasked_span):
                    span_label_tokens[j] = True

    assert len(masked_spans) <= num_to_predict
    masked_spans = sorted(masked_spans, key=lambda x: x.index[0])

    # convert to pythonic way of span representation
    question_spans = [(s.index[0], s.index[1] + 1) for s in masked_spans]
    label_spans = [(s.begin_label, s.end_label + 1) for s in masked_spans]

    return new_tokens, question_spans, label_spans, span_clusters


def is_interesting_span(span):
    """
    :param span: list of words which are part of the span
    :return: bool indicating if span is interesting for further processing
    """
    words = span[-1]
    first: str = words[0]
    if len(words) > 1:
        return True
    # long enough words
    elif len(first) > 6:
        return True
    # shorter words which are either numeric and not lowercased
    elif len(first) > 3 and (first.isnumeric() or first.lower() != first):
        return True
    else:
        return False


@feed_single_example
def find_recurring_spans(example, tokenizer, max_questions=10, **kwargs):
    """
    :param example: example after chunking process
    :param tokenizer: tokenizer
    :param max_questions: maximum number of questions in the training/inference example. For docuemnts with
    many entities multiple examples will be yielded
    :param kwargs:
    :return: new example with additional fields
    """
    words, chunk_range, word_map, word_starts, input_ids, content_tokens_mask = (
        example["words"],
        example["chunk_ranges"],
        example["word_map"],
        example["word_starts"],
        example["input_ids"],
        example["content_tokens_mask"],
    )
    word_starts = example["word_starts"]
    chunk_words = words[word_map[0] : word_map[-1] + 1]
    new_tokens, question_spans, label_spans, span_clusters = create_recurring_span_selection_predictions(
        chunk_words, max_questions, 10, 0.15
    )

    if len(question_spans) < 3:
        return None

    word_map_with_offset = word_map - word_map[0]
    question_token_spans = recalculate_spans([question_spans], [word_map_with_offset])[0]
    label_token_spans = recalculate_spans([label_spans], [word_map_with_offset])[0]

    input_ids = np.array(input_ids)
    input_ids_content = input_ids[content_tokens_mask]
    attention_mask = np.array(example["attention_mask"])
    attention_mask_content = attention_mask[content_tokens_mask]

    if hasattr(tokenizer, "question_token_id"):
        question_token_id = tokenizer.question_token_id
    else:
        question_token_id = tokenizer.mask_token_id

    answer_token_label_ids_lst = []
    question_token_positions = []
    context_start = np.where(content_tokens_mask)[0][0]

    for (q1, q2), (l1, l2) in zip(question_token_spans, label_token_spans):
        lab = np.zeros((len(attention_mask_content)))

        lab[l1:l2] = 1

        input_ids_content[q1] = question_token_id
        attention_mask_content[q1 + 1 : q2] = False
        # TODO: add +1 to these positions for CLS
        question_token_positions.append(q1)
        answer_token_label_ids_lst.append(lab)

    answer_token_label_ids_cont = np.stack(answer_token_label_ids_lst, -1)
    answer_token_label_ids_cont[np.logical_not(word_starts)] = -100
    answer_token_label_ids = fill_special_tokens(answer_token_label_ids_cont, content_tokens_mask, -100)
    pad_len = max_questions - len(question_token_spans)
    answer_token_label_ids_padded = np.pad(answer_token_label_ids, ((0, 0), (0, pad_len)), constant_values=-100)
    input_ids[content_tokens_mask] = input_ids_content
    attention_mask[content_tokens_mask] = attention_mask_content
    question_token_positions = [pos + context_start for pos in question_token_positions]

    return {
        **example,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "question_positions": question_token_positions,
        "answer_token_label_ids": answer_token_label_ids_padded,
    }


@feed_single_example_and_flatten
def build_prefix_with_mqa_splinter(example, tokenizer, **kwargs):
    """
    @param example: document features
    @param tokenizer: used tokenizer, have to contain mask_token or question_token
    @param max_questions: max questions to be generated for given doc, if there are more we generate separate examples
        for remaining questions
    @param kwargs:
    @return: example with additional features specific to splinter
    """
    # max questions to be generated for given doc, if there are more we generate separate examples
    # for remaining questions
    max_questions = SPLINTER_MAX_QUESTIONS
    entities = example["entities"]
    if hasattr(tokenizer, "question_token_id"):
        question_token = tokenizer.question_token
    else:
        question_token = tokenizer.mask_token
    lst_entities = convert_to_list_of_dicts(entities)
    # compute how many entities should be in single example
    ent_ranges = get_chunk_ranges(len(lst_entities), chunk_size=max_questions, overlap=0)

    for fr_range, to_range in ent_ranges:

        ex_entities = lst_entities[fr_range:to_range]

        prefixes = []
        question_positions = []
        answer_token_label_ids_lst = []
        for idx, entity in enumerate(ex_entities):
            prefix = [entity["name"]] + [question_token]
            question_positions.append(2 * idx + 1)
            ans_ids = np.zeros((len(example["words"])), dtype=np.int)
            for span in entity["token_spans"]:
                ans_ids[span[0] : span[1]] = 1
            answer_token_label_ids_lst.append(ans_ids)
            prefixes.extend(prefix)
        prefixes = prefixes + [tokenizer.sep_token]
        answer_token_label_ids = np.stack(answer_token_label_ids_lst, -1)
        pad_len = max_questions - len(answer_token_label_ids_lst)
        answer_token_label_ids_padded = np.pad(answer_token_label_ids, ((0, 0), (0, pad_len)), constant_values=-100)

        yield {
            **example,
            "entities": convert_to_dict_of_lists(ex_entities, keys=list(entities.keys())),
            "question_positions": question_positions,
            "answer_token_label_ids": answer_token_label_ids_padded,
            "prefix_words": prefixes,
        }
