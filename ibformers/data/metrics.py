from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
from datasets import load_metric, Dataset


def get_predictions_for_sl(predictions: Tuple, dataset: Dataset):
    features = dataset.features
    assert 'id' in features, 'dataset need to contain ids of documents'
    label_list = features['labels'].feature.names

    preds, labels = predictions

    ids = dataset['id']
    assert len(set(ids)) == len(ids), 'chunks are not supported by this function'
    # TODO: add chunk support, use for that chunk offsets stored in ds features

    # softmax for np
    pred_prob = np.exp(preds) / np.sum(np.exp(preds), axis=-1, keepdims=True)
    pred_conf = np.max(pred_prob, axis=-1)
    pred_class_index = np.argmax(pred_prob, axis=-1)

    # temporary fix for old version of transformers in IB
    dataset._output_all_columns = True

    predictions = {}

    for doc_conf, doc_class_index, doc_lab, doc in zip(pred_conf, pred_class_index, labels, dataset):
        # remove padding
        inp_len = len(doc['attention_mask'])
        doc_conf, doc_class_index, doc_lab = doc_conf[:inp_len], doc_class_index[:inp_len], doc_lab[:inp_len]
        # remove special tokens
        non_special = np.logical_not(np.array(doc['special_tokens_mask']))
        doc_conf, doc_class_index, doc_lab = doc_conf[non_special], doc_class_index[non_special], doc_lab[non_special]
        # get word level predictions - we might want to change that to support token level predictions
        # word-begin indices
        word_indices = np.array(doc['offset_mapping'])[:, 0] == 0
        doc_conf, doc_class_index, doc_lab = doc_conf[word_indices], doc_class_index[word_indices], doc_lab[
            word_indices]
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
        non_zero_golden_class = np.nonzero(doc_lab)[0]
        for idx in non_zero_golden_class:
            class_idx = doc_lab[idx]
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

        predictions[doc['id']] = doc_dict

    dataset._output_all_columns = False

    return predictions


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
    print(pd.DataFrame(final_results))

    # get prediction dict and print mismatches
    predictions = get_predictions_for_sl(predictions, dataset)

    print("MISMATCH EXAMPLES")
    max_examples = 2
    for lab in label_list[1:]:
        mismatches = ["\tpred:\t'" + v[lab]['text'] + "'\n\tgold:\t'" + v[lab]['gold'] + "'\n"
                      for k, v in predictions.items() if not v[lab]['is_match']]
        mismatch_text = '  '.join(mismatches[:max_examples])
        if len(mismatches) > 0:
            print(f"{lab}:\n{mismatch_text}", end="")

    final_results['predictions'] = predictions

    return final_results
