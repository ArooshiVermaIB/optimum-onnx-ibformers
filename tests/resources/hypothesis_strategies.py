import bisect
from typing import Tuple, List, Optional, Sequence, Dict, Any

import datasets
from hypothesis import strategies as st

from ibformers.data.utils import ImageProcessor
from ibformers.datasets.utils import normalize_bbox

BBOX_MAX_ABS_VALUE = 4096
MAX_WIDTH_HEIGTH = int(0.1 * BBOX_MAX_ABS_VALUE)
MAX_ALLOWED_OUT_OF_BOUNDS = int(0.2 * BBOX_MAX_ABS_VALUE)
Bbox = Tuple[int, int, int, int]


def _get_matching_page_bboxes(page_bboxes: List[Bbox], page_spans: List[Tuple[int, int]]) -> List[Bbox]:
    span_ends = [page_span[1] for page_span in page_spans]
    return [page_bboxes[bisect.bisect_right(span_ends, i)] for i in range(page_spans[-1][-1])]


def _normalize_bboxes(
    original_bboxes: List[Bbox],
    page_bboxes: List[Bbox],
    page_spans: List[Tuple[int, int]],
):
    matching_page_bboxes = _get_matching_page_bboxes(page_bboxes, page_spans)
    return [
        normalize_bbox(original_bbox, matching_page_bbox[2:4])
        for original_bbox, matching_page_bbox in zip(original_bboxes, matching_page_bboxes)
    ]


@st.composite
def bbox(draw, allow_invalid_bboxes: bool, max_bbox: Optional[Bbox] = None) -> Bbox:
    if max_bbox is None:
        max_bbox = [0, 0, BBOX_MAX_ABS_VALUE, BBOX_MAX_ABS_VALUE]
    max_x = max_bbox[2] + MAX_ALLOWED_OUT_OF_BOUNDS if allow_invalid_bboxes else max_bbox[2]
    max_y = max_bbox[3] + MAX_ALLOWED_OUT_OF_BOUNDS if allow_invalid_bboxes else max_bbox[3]
    min_x = max_bbox[0] - MAX_ALLOWED_OUT_OF_BOUNDS if allow_invalid_bboxes else max_bbox[0]
    min_y = max_bbox[1] - MAX_ALLOWED_OUT_OF_BOUNDS if allow_invalid_bboxes else max_bbox[1]

    x_start = draw(st.integers(min_value=min_x, max_value=max_x))
    y_start = draw(st.integers(min_value=min_y, max_value=max_y))

    min_hw = -MAX_WIDTH_HEIGTH if allow_invalid_bboxes else 0
    max_hw = MAX_WIDTH_HEIGTH
    height = draw(st.integers(min_value=min_hw, max_value=max_hw))
    width = draw(st.integers(min_value=min_hw, max_value=max_hw))

    x_end = x_start + width if allow_invalid_bboxes else min(max_x, x_start + width)
    y_end = y_start + height if allow_invalid_bboxes else min(max_y, y_start + height)

    return x_start, y_start, x_end, y_end


@st.composite
def entity_data(draw, num_fields: int, generated_words: Sequence[str]):
    entity_names = ["O"] + [f"e{i+1}" for i in range(num_fields)]
    example_len = len(generated_words)
    entity_dict = {
        name: {
            "name": name,
            "order_id": 0,
            "text": "",
            "char_spans": [],
            "token_spans": [],
            "token_label_id": entity_names.index(name),
        }
        for name in entity_names[1:]
    }
    token_label_ids = draw(
        st.lists(st.integers(min_value=0, max_value=num_fields), min_size=example_len, max_size=example_len)
    )
    for i, (token_text, token_label) in enumerate(zip(generated_words, token_label_ids)):
        if token_label == 0:
            continue
        entity_name = entity_names[token_label]
        entity_dict[entity_name]["text"] += f" {token_text}"
        token_spans = entity_dict[entity_name]["token_spans"]
        if len(token_spans) == 0 or token_spans[-1][-1] != i:
            token_spans.append([i, i + 1])
        else:
            token_spans[-1][-1] += 1
    return {"token_label_ids": token_label_ids, "entities": list(entity_dict.values())}


@st.composite
def example(
    draw,
    min_example_len: int,
    max_example_len: int,
    num_fields: int,
    allow_invalid_bboxes: bool = False,
    allowed_text_characters: Optional[str] = None,
):
    example_len = draw(st.integers(min_value=min_example_len, max_value=max_example_len))
    if allowed_text_characters is None:
        words = draw(st.lists(st.text(min_size=1, max_size=10), min_size=example_len, max_size=example_len))
    else:
        words = draw(
            st.lists(
                st.text(allowed_text_characters, min_size=1, max_size=10), min_size=example_len, max_size=example_len
            )
        )

    features = {
        "id": draw(st.text(min_size=1, max_size=10)),
        "words": words,
        "word_original_bboxes": ([],),
        "word_page_nums": [0] * example_len,
        "page_bboxes": [draw(bbox(allow_invalid_bboxes))],
        "page_spans": [(0, example_len)],
        "images": ImageProcessor().get_default_processed_image()[None, :],
        "images_page_nums": [0],
    }
    features["word_original_bboxes"] = [
        draw(bbox(allow_invalid_bboxes, page_bbox))
        for page_bbox in _get_matching_page_bboxes(features["page_bboxes"], features["page_spans"])
    ]
    features["bboxes"] = _normalize_bboxes(
        features["word_original_bboxes"], features["page_bboxes"], features["page_spans"]
    )
    entity_dict = draw(entity_data(num_fields, features["words"]))
    features.update(entity_dict)
    return features


def create_dataset_from_examples(examples: List[Dict[str, Any]]):
    feature_names = ["O"] + list([entity_dict["name"] for entity_dict in examples[0]["entities"]])
    ds_features = {
        "id": datasets.Value("string"),
        "words": datasets.Sequence(datasets.Value("string")),
        "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
        "word_original_bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("float32"), length=4)),
        "word_page_nums": datasets.Sequence(datasets.Value("int32")),
        "page_bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=4)),
        "page_spans": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=2)),
        "token_label_ids": datasets.Sequence(datasets.features.ClassLabel(names=feature_names)),
        "entities": datasets.Sequence(
            {
                "name": datasets.Value("string"),  # change to id?
                "order_id": datasets.Value("int64"),
                # not supported yet, annotation app need to implement it
                "text": datasets.Value("string"),
                "char_spans": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=2)),
                "token_spans": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=2)),
                "token_label_id": datasets.Value("int64"),
            }
        ),
        "images": datasets.Array4D(shape=(None, 3, 224, 224), dtype="uint8"),
        "images_page_nums": datasets.Sequence(datasets.Value("int32")),
    }
    features = datasets.Features(ds_features)
    data = {key: [data_example[key] for data_example in examples] for key in examples[0].keys()}
    return datasets.Dataset.from_dict(data, features)
