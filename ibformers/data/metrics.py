from collections import defaultdict
from typing import Tuple, Iterator, Union, List, Sequence

import numpy as np
import pandas as pd
from datasets import load_metric, Dataset


def doc_chunk_iter(doc_ids: List[str]) -> Iterator[Tuple[str, int, int]]:
    """
    function will return list of doc_ids with ranges of chunks corresponding to given doc_id
    :param doc_ids: list of document ids
    :return: Iterator of tuples
    """
    from_idx = 0
    next_doc_ids = doc_ids[1:] + ["*end*"]
    for i, (doc_id, next_doc_id) in enumerate(zip(doc_ids, next_doc_ids)):
        if doc_id != next_doc_id:
            yield doc_id, from_idx, i+1
            from_idx = i+1


def join_chunks(chunks: Union[List[Sequence], np.ndarray], chunk_ranges: List[Sequence[int, int]]) -> np.ndarray:
    """
    When we get predictions for overlapping chunks of an input sequence, we have to combine the predictions, doing
    something with the overlap. In this function, we simply take the feature-wise mean for each of the tokens that
    have predictions from multiple chunks.
    Join list of ndarray of chunks based on the information in the chunk_ranges list
    :param chunks: Chunks which need to be joined
    :param chunk_ranges: Sequence of ranges which inform about position of chunk in the original document
    :return:
    """
    if len(chunks) == 1:
        rng = chunk_ranges[0]
        return np.array(chunks[0])[rng[0]:rng[1]]
    strictly_increasing = all(i[0] < j[0] for i, j in zip(chunk_ranges, chunk_ranges[1:]))
    assert strictly_increasing, f"Ranges of the chunks seems incorrect: Value: {chunk_ranges}"
    max_size = chunk_ranges[-1][-1]
    first_chunk = np.array(chunks[0])
    chunk_shape = first_chunk.shape
    doc_arr = np.full((len(chunks), max_size, *chunk_shape[1:]), fill_value=np.nan)
    for i, (chunk, rng) in enumerate(zip(chunks, chunk_ranges)):
        rng_len = rng[1]-rng[0]
        doc_arr[i, rng[0]:rng[1]] = chunk[:rng_len]

    doc_arr_mean = np.nanmean(doc_arr, axis=0)

    return doc_arr_mean.astype(first_chunk.dtype)


def get_predictions_for_sl(predictions: Tuple, dataset: Dataset):
    features = dataset.features
    assert 'id' in features, 'dataset need to contain ids of documents'
    label_list = features['labels'].feature.names
    preds, labels = predictions
    ids = dataset['id']
    chunk_ranges = dataset['chunk_ranges']
    pred_dict = {}

    for doc_id, chunk_from_idx, chunk_to_idx in doc_chunk_iter(ids):
        doc = dataset[chunk_from_idx]
        assert doc_id == doc['id'], "Chunk doc_id and doc_id obtained from the dataset does not match"

        chunk_ranges_lst = chunk_ranges[chunk_from_idx: chunk_to_idx]
        # get rid of CLS token and last token for preds and labels
        preds_arr = preds[chunk_from_idx: chunk_to_idx, 1:-1]
        labels_arr = labels[chunk_from_idx: chunk_to_idx, 1:-1]
        offset_mapping_lst = dataset['offset_mapping'][chunk_from_idx:chunk_to_idx]

        doc_preds = join_chunks(preds_arr, chunk_ranges_lst)
        doc_labels = join_chunks(labels_arr, chunk_ranges_lst)
        doc_offsets = join_chunks(offset_mapping_lst, chunk_ranges_lst)
        word_indices = np.array(doc_offsets)[:, 0] == 0


        # softmax for np
        doc_prob = np.exp(doc_preds) / np.sum(np.exp(doc_preds), axis=-1, keepdims=True)
        doc_conf = np.max(doc_prob, axis=-1)
        doc_class_index = np.argmax(doc_prob, axis=-1)

        # get word level predictions - we might want to change that to support token level predictions
        # word-begin indices

        doc_conf = doc_conf[word_indices]
        doc_class_index = doc_class_index[word_indices]
        doc_labels = doc_labels[word_indices]

        non_zero_class = np.nonzero(doc_class_index)[0]
        doc_words_dict = defaultdict(list)
        for idx in non_zero_class:
            class_idx = doc_class_index[idx]
            conf = doc_conf[idx]
            tag_name = label_list[class_idx]
            org_bbox = doc['word_original_bboxes'][idx]
            word = dict(word=doc['words'][idx],
                        start_x=org_bbox[0], start_y=org_bbox[1], end_x=org_bbox[2], end_y=org_bbox[3],
                        conf=conf,
                        idx=idx)
            doc_words_dict[tag_name].append(word)

        # generate correct answers to print pred/gold mismatches
        golden_words_dict = defaultdict(list)
        non_zero_golden_class = np.nonzero(doc_labels > 0)[0]
        for idx in non_zero_golden_class:
            class_idx = doc_labels[idx]
            tag_name = label_list[class_idx]
            golden_words_dict[tag_name].append(doc['words'][idx])

        doc_dict = {}
        for k in label_list[1:]:
            pred_words = doc_words_dict.get(k, [])
            pred_text = ' '.join([w['word'] for w in pred_words])
            golden = ' '.join(golden_words_dict.get(k, []))
            doc_dict[k] = {'words': pred_words,
                           'text': pred_text,
                           'avg_confidence': np.mean([w['conf'] for w in pred_words]),
                           'gold': golden,
                           'is_match': pred_text == golden}

        pred_dict[doc['id']] = doc_dict

    return pred_dict


def compute_metrics_for_sl(predictions: Tuple, dataset: Dataset):
    metric = load_metric("seqeval")
    preds, labels = predictions
    pred_class_index = np.argmax(preds, axis=-1)

    features = dataset.features
    label_list = features['labels'].feature.names

    # add tags to label_list
    label_list_tags = [f'I-{label}' if label != 'O' else 'O' for label in label_list]

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list_tags[p] for (p, l) in zip(doc_pred_cls, label) if l != -100]
        for doc_pred_cls, label in zip(pred_class_index, labels)
    ]
    true_labels = [
        [label_list_tags[l] for (p, l) in zip(doc_pred_cls, label) if l != -100]
        for doc_pred_cls, label in zip(pred_class_index, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    # Unpack nested dictionaries
    final_results = {'precision': {},
                     'recall': {},
                     'f1': {}}

    for key, value in results.items():
        if isinstance(value, dict):
            final_results['precision'][key] = value['precision']
            final_results['recall'][key] = value['recall']
            final_results['f1'][key] = value['f1']
    final_results['precision']['_Overall'] = results["overall_precision"]
    final_results['recall']['_Overall'] = results["overall_recall"]
    final_results['f1']['_Overall'] = results["overall_f1"]
    print("EVALUATION RESULTS")
    print(pd.DataFrame(final_results))

    # get prediction dict and print mismatches
    pred_dict = get_predictions_for_sl(predictions, dataset)

    print("MISMATCH EXAMPLES")
    max_examples = 2
    for lab in label_list[1:]:
        mismatches = ["\tpred:\t'" + v[lab]['text'] + "'\n\tgold:\t'" + v[lab]['gold'] + "'\n"
                      for k, v in pred_dict.items() if not v[lab]['is_match']]
        mismatch_text = '  '.join(mismatches[:max_examples])
        if len(mismatches) > 0:
            print(f"{lab}:\n{mismatch_text}", end="")

    final_results['predictions'] = pred_dict

    return final_results
