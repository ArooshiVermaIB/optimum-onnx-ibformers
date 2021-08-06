import json
import logging
from typing import Tuple, List, Dict, Any, Union, NamedTuple, Optional

import datasets
from pathlib import Path
from dataclasses import dataclass
from instabase.ocr.client.libs.ibocr import ParsedIBOCRBuilder, IBOCRRecord, IBOCRRecordLayout
from instabase.ocr.client.libs.ocr_types import WordPolyDict
from typing_extensions import TypedDict, Literal
from more_itertools import consecutive_groups
import numpy as np


_DESCRIPTION = """\
Internal Instabase Dataset format organized into set of IbDoc files and IbAnnotator file.
"""


BoundingBox = Tuple[float, float, float, float]
Span = Tuple[int, int]


@dataclass()
class OcrDoc:
    doc_id: str
    words: List[str]
    bboxes: List[BoundingBox]
    page_bboxes: List[BoundingBox]
    page_spans: List[Span]


def norm_bbox(bbox, width):
    # divide only by width to keep information about ratio
    return (
        bbox[0] / width,
        bbox[1] / width,
        bbox[2] / width,
        bbox[3] / width,
    )


def read_ibdoc(ibdoc_path, ibsdk=None):
    """Open an ibdoc or ibocr using the ibfile and return the words and layout information for each page"""
    open_fn = open if ibsdk is None else ibsdk.ibopen
    data = open_fn(ibdoc_path, 'rb').read()
    builder: ParsedIBOCRBuilder
    builder, err = ParsedIBOCRBuilder.load_from_str(ibdoc_path, data)
    if err:
        raise IOError(u'Could not load file: {}'.format(ibdoc_path))

    words = []
    layouts = []
    record: IBOCRRecord
    # Assuming each record is a page in order and each record is single-page
    # Assuming nothing weird is going on with page numbers
    for record in builder.get_ibocr_records():
        words += [i for j in record.get_lines() for i in j]
        l = record.get_metadata_list()
        layouts.extend([i.get_layout() for i in l])

    assert all(
        word['page'] in range(len(layouts)) for word in words
    ), "Something with the page numbers went wrong"

    return words, layouts


AnnotationLabelId = str


class AnnotationLayout(TypedDict):
    width: float
    height: float


class AnnotationRect(TypedDict):
    x: int
    y: int
    w: int
    h: int


class AnnotationLabel(TypedDict):
    id: str
    name: str
    type: str


class AnnotationPosition(TypedDict):
    rect: AnnotationRect
    page: int


class AnnotationWordMetadata(TypedDict):
    dataType: Literal["WordPoly"]
    type: Literal['rect']
    rawWord: str
    position: AnnotationPosition


class Annotation(TypedDict):
    value: str
    metadata: List[Union[AnnotationWordMetadata, Any]]


class AnnotationFile(TypedDict):
    id: str
    inputPath: str
    ocrPath: str
    isInputOcr: bool
    pages: Dict[str, AnnotationLayout]
    annotations: Dict[AnnotationLabelId, Annotation]


class Annotation(TypedDict):
    value: str
    metadata: List[Union[AnnotationWordMetadata, Any]]


class _SearchKey(NamedTuple):
    x: float
    y: float
    page: int
    word: str


def process_example(
    id2label: Dict[str, str],
    doc_annotations: AnnotationFile,
    words: List[WordPolyDict],
    layouts: List[IBOCRRecordLayout],
    str2int: Dict[str, int]
):

    word_lst = [w['word'] for w in words]
    bbox_arr = np.array([[w['start_x'], w['start_y'], w['end_x'], w['end_y']] for w in words])
    word_pages_arr = np.array([w['page'] for w in words])

    # get number of tokens for each page, below is a bit overcomplicated because there might be empty pages
    page_nums, page_token_counts = np.unique(word_pages_arr, return_counts=True)
    page_ct_arr = np.zeros(len(layouts), dtype=np.int64)  # assume that layouts are single pages
    for page_num, page_token_ct in zip(page_nums, page_token_counts):
        page_ct_arr[page_num] = page_token_ct

    # get page spans
    page_offsets = np.cumsum(page_ct_arr)
    page_spans = np.stack((np.pad(page_offsets[:-1], (1, 0)), page_offsets), axis=-1)

    # get page height and width
    page_bboxes = np.array([[0, 0, pg.get_width(), pg.get_height()] for pg in layouts])

    # normalize bboxes - divide only by width to keep information about ratio
    width_per_token = np.take(page_bboxes[:, 2], word_pages_arr)
    norm_bboxes = bbox_arr / width_per_token[:, None]
    norm_page_bboxes = page_bboxes / page_bboxes[:, 2:3]

    entities = []
    annotation: Annotation
    token_label_ids = np.zeros((len(word_lst)), dtype=np.int64)
    for label_id, annotation in doc_annotations['annotations'].items():
        metadata = annotation['metadata']
        label_name = id2label[label_id]
        word_metadata: AnnotationWordMetadata
        label_words = []
        for word_metadata in metadata:
            if word_metadata['dataType'] != 'WordPoly':
                raise ValueError("Unexpected (non-wordpoly) annotation. Skipping.")
            word_id_in_page = int(word_metadata['appData']['id']) - 1  # switch to 0-indexing
            page_num = word_metadata['position']['page']
            word_id_global = word_id_in_page + page_spans[page_num, 0]
            word_text = word_metadata['rawWord']
            assert word_text.strip() == words[word_id_global]['word'].strip(), \
                "Annotation does not match with document words"

            label_words.append({'id': word_id_global, 'text': word_text})

        # TODO: information about multi-item entity separation should be obtained during annotation
        # group consecutive words as the same entity occurrence
        label_words.sort(key=lambda x: x["id"])
        label_groups = [list(group) for group in consecutive_groups(label_words, ordering=lambda x: x["id"])]
        # create spans for groups, span will be created by getting id for first and last word in the group
        label_token_spans = [[group[0]["id"], group[-1]["id"] + 1] for group in label_groups]

        for span in label_token_spans:
            token_label_ids[span[0]:span[1]] = str2int[label_name]

        entity = {'name': label_name,
                  'order_id': 0,
                  'text': annotation['value'],
                  'char_spans': [],
                  'token_spans': label_token_spans}
        entities.append(entity)

    return {
        "id": doc_annotations["id"],
        "words": word_lst,
        "bboxes": norm_bboxes,
        "page_bboxes": norm_page_bboxes,
        "page_spans": page_spans,
        'token_label_ids': token_label_ids,
        "entities": entities,
        }


class IbDsConfig(datasets.BuilderConfig):
    """
    Config for Instabase Format Datasets
    """

    def __init__(self, use_image=False, ibsdk=None, **kwargs):
        """BuilderConfig for Instabase Format Datasets.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(IbDsConfig, self).__init__(**kwargs)
        self.use_image = use_image
        self.ibsdk = ibsdk


class IbDs(datasets.GeneratorBasedBuilder):
    """TODO(ibds): Short description of my dataset."""

    BUILDER_CONFIGS = [
        IbDsConfig(name="ibds", version=datasets.Version("1.0.0"), description="Instabase Format Datasets"),
    ]

    def _info(self):

        # get schema of the the dataset
        # TODO(ibds): Check if schema can be saved in the separate file, so we don't load whole annotation file
        data_files = self.config.data_files
        assert len(data_files) == 1, 'Only one annotation path should be provided'
        with open(data_files[0], 'r', encoding='utf-8') as annotation_file:
            labels = json.load(annotation_file)["labels"]

        self.id2label = {lab["id"]: lab["name"] for lab in labels}
        classes = ['O'] + list(self.id2label.values())

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    'id': datasets.Value('string'),
                    'words': datasets.Sequence(datasets.Value('string')),
                    'bboxes': datasets.Sequence(datasets.Sequence(datasets.Value('float'), length=4)),
                    'page_bboxes': datasets.Sequence(datasets.Sequence(datasets.Value('float'), length=4)),
                    'page_spans': datasets.Sequence(datasets.Sequence(datasets.Value('int64'), length=2)),
                    'token_label_ids': datasets.Sequence(datasets.features.ClassLabel(names=classes)),
                    'entities': datasets.Sequence(
                        {
                            'name': datasets.Value('string'),  # change to id?
                            'order_id': datasets.Value('int64'),  # not supported yet, annotation app need to implement it
                            'text': datasets.Value('string'),
                            'char_spans': datasets.Sequence(datasets.Sequence(datasets.Value('int64'), length=2)),
                            'token_spans': datasets.Sequence(datasets.Sequence(datasets.Value('int64'), length=2))
                        }
                    ),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        data_files = self.config.data_files
        with open(data_files[0], 'r', encoding='utf-8') as annotation_file:
            annotations = json.load(annotation_file)

        ibsdk = self.config.ibsdk

        self.id2label = {lab["id"]: lab["name"] for lab in annotations["labels"]}

        # create generators for train and test
        train_files = (file for file in annotations['files'] if file['id'] not in annotations['testFiles'])
        test_files = (file for file in annotations['files'] if file['id'] in annotations['testFiles'])

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={'files': train_files, 'ibsdk': ibsdk}),
            datasets.SplitGenerator(name=datasets.Split.TEST,
                                    gen_kwargs={'files': test_files, 'ibsdk': ibsdk}),
        ]

    def _generate_examples(self, files, ibsdk):
        """Yields examples."""
        for file in files:
            ocr_path = file['ocrPath']
            if not (ocr_path.endswith('.ibdoc') or ocr_path.endswith('.ibocr')):
                raise ValueError(f"Invaild document path: {ocr_path}")

            words, layouts = read_ibdoc(ocr_path, ibsdk)
            str2int = self.info.features['token_label_ids'].feature._str2int
            doc_dict = process_example(self.id2label, file, words, layouts, str2int)

            yield file["id"], doc_dict
