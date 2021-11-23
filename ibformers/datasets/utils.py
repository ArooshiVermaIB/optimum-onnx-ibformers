from collections import defaultdict
from typing import List, Dict, Any, Tuple
import numpy as np


def normalize_bbox(bbox: Tuple[int, int, int, int], size: Tuple[int, int]):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def enrich_features_with_images(features, image, image_processor):
    word_pages_arr = features['word_page_nums']
    page_nums = np.unique(word_pages_arr)
    page_nums.sort()
    image_postprocessed = image_processor.postprocess(image)
    # assert len(norm_page_bboxes) == len(images), "Number of images should match number of pages in document"
    features["images"] = [image_postprocessed]
    features["images_page_nums"] = page_nums


def create_features_from_file_content(
    file_contents: List[Dict[str, Any]], image_size: Tuple[int, int], label2id: Dict[str, int]
) -> Dict[str, Any]:

    features = defaultdict(list)
    features['page_bboxes'].append([0, 0, *image_size])

    entity_dicts = {
        label: {
            'name': label,
            'order_id': 0,
            'text': '',
            'char_spans': [],
            'token_label_id': label2id[label],
            'token_spans': [],
        }
        for label in label2id.keys()
        if label != 'O'
    }

    word_id_counter = 0
    for item in file_contents:
        words_example, label = item["words"], item["label"]
        label = label.upper()
        words_example = [w for w in words_example if w["text"].strip() != ""]
        if len(words_example) == 0:
            continue
        if label == "OTHER":
            for w in words_example:
                features['words'].append(w["text"])
                features['token_label_ids'].append("O")
                features['bio_token_label_ids'].append("O")
                features['bboxes'].append(normalize_bbox(w["box"], image_size))
                features['word_original_bboxes'].append(w["box"])
                features['word_page_nums'].append(0)
                word_id_counter += 1
        else:
            entity_dict = entity_dicts[label]
            entity_text = ' '.join([w['text'] for w in words_example])
            start_word_id = word_id_counter
            features['words'].append(words_example[0]["text"])
            features['token_label_ids'].append(label)
            features['bio_token_label_ids'].append("B-" + label)
            features['bboxes'].append(normalize_bbox(words_example[0]["box"], image_size))
            features['word_original_bboxes'].append(words_example[0]["box"])
            features['word_page_nums'].append(0)
            word_id_counter += 1
            for w in words_example[1:]:
                features['words'].append(w["text"])
                features['token_label_ids'].append(label)
                features['bio_token_label_ids'].append("I-" + label)
                features['bboxes'].append(normalize_bbox(w["box"], image_size))
                features['word_original_bboxes'].append(w["box"])
                features['word_page_nums'].append(0)
                word_id_counter += 1
            entity_dict['token_spans'].append([start_word_id, word_id_counter])
            entity_dict['text'] += f' {entity_text}'
    features['entities'] = list(entity_dicts.values())
    return features
