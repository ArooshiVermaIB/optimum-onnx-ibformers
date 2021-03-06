from collections import defaultdict
from typing import List, Dict, Any, Tuple

import numpy as np
from typing_extensions import TypedDict


def normalize_bbox(bbox: Tuple[int, int, int, int], size: Tuple[int, int]):
    zero_width = size[0] == 0
    zero_height = size[1] == 0
    return [
        int(1000 * bbox[0] / size[0]) if not zero_width else 0,
        int(1000 * bbox[1] / size[1]) if not zero_height else 0,
        int(1000 * bbox[2] / size[0]) if not zero_width else 0,
        int(1000 * bbox[3] / size[1]) if not zero_height else 0,
    ]


def enrich_features_with_images(features, image, image_processor):
    word_pages_arr = features["word_page_nums"]
    page_nums = np.unique(word_pages_arr)
    page_nums.sort()
    image_postprocessed = image_processor.postprocess(image)
    # assert len(norm_page_bboxes) == len(images), "Number of images should match number of pages in document"
    features["images"] = image_postprocessed[None, :]
    features["images_page_nums"] = page_nums


class FundWordRecord(TypedDict):
    """
    Attributes:
        text: text of the word
        bbox: bounding box of the word
    """

    text: str
    box: Tuple[int, int, int, int]


class FundEntityRecord(TypedDict):
    """
    Attributes:
        id: the id of the entity
        text: the text of the entity
        bbox: the bounding boxcovering the whole entity. Note that the entity might span over whole paragraphs.
        linking: list of the relations that this entity is a part of
        label: entity label name
        words: list of word data
    """

    id: int
    text: str
    box: Tuple[int, int, int, int]
    linking: List[Tuple[int, int]]
    label: str
    words: List[FundWordRecord]


def get_line_bbox(bboxs):
    x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
    y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

    assert x1 >= x0 and y1 >= y0
    bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
    return bbox


def create_features_from_fund_file_content(
    file_contents: List[FundEntityRecord],
    image_size: Tuple[int, int],
    label2id: Dict[str, int],
    consider_line_segments: bool = False,
) -> Dict[str, Any]:
    """
    Extract features from FUNSD/XFUND documents data.

    Args:
        file_contents: List of dictionaries with document data.
        image_size: Tuple of page dimensions, used for bbox normalization
        label2id: mapping of label names to label ids
        consider_line_segments: whether to consider the bbox to be line segment level
        instead of indiviual word level. This has shown to improve the f1 score, and is used in layoutlmv3 like here:
        https://github.com/microsoft/unilm/blob/925de7a9ea500e992ec5de02ea193a5eb9d5aa26/layoutlmv3/layoutlmft/data/funsd.py#L129
    Returns:
        Processed features.
    """

    features = defaultdict(list)
    features["page_bboxes"].append(np.array([0, 0, *image_size]))

    entity_dicts = {
        label: {
            "name": label,
            "order_id": 0,
            "text": "",
            "char_spans": [],
            "token_label_id": label2id[label],
            "token_spans": [],
        }
        for label in label2id.keys()
        if label != "O"
    }

    word_id_counter = 0
    for item in file_contents:
        cur_line_bboxes = []
        words_example, label = item["words"], item["label"]
        label = label.upper()
        words_example = [w for w in words_example if w["text"].strip() != ""]
        if len(words_example) == 0:
            continue
        if label == "OTHER":
            for w in words_example:
                features["words"].append(w["text"])
                features["token_label_ids"].append("O")
                features["bio_token_label_ids"].append("O")
                cur_line_bboxes.append(normalize_bbox(w["box"], image_size))
                features["word_original_bboxes"].append(w["box"])
                features["word_page_nums"].append(0)
                word_id_counter += 1
        else:
            entity_dict = entity_dicts[label]
            entity_text = " ".join([w["text"] for w in words_example])
            start_word_id = word_id_counter
            features["words"].append(words_example[0]["text"])
            features["token_label_ids"].append(label)
            features["bio_token_label_ids"].append("B-" + label)
            cur_line_bboxes.append(normalize_bbox(words_example[0]["box"], image_size))
            features["word_original_bboxes"].append(words_example[0]["box"])
            features["word_page_nums"].append(0)
            word_id_counter += 1
            for w in words_example[1:]:
                features["words"].append(w["text"])
                features["token_label_ids"].append(label)
                features["bio_token_label_ids"].append("I-" + label)
                cur_line_bboxes.append(normalize_bbox(w["box"], image_size))
                features["word_original_bboxes"].append(w["box"])
                features["word_page_nums"].append(0)
                word_id_counter += 1
            entity_dict["token_spans"].append([start_word_id, word_id_counter])
            entity_dict["text"] += f" {entity_text}"
        if consider_line_segments:
            cur_line_bboxes = get_line_bbox(cur_line_bboxes)
        features["bboxes"].extend(cur_line_bboxes)
    features["entities"] = list(entity_dicts.values())

    features["bboxes"] = np.array(features["bboxes"])
    features["word_original_bboxes"] = np.array(features["word_original_bboxes"])

    return features
