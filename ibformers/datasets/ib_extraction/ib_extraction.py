import logging
import os
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Sequence, NamedTuple, Iterable, Any

import datasets
import numpy as np
from more_itertools import consecutive_groups
from typing_extensions import TypedDict

from ibformers.data.utils import ImageProcessor
from ibformers.datasets.ib_common.ib_common import (
    assert_valid_record,
    Span,
    load_datasets,
    get_common_feature_schema,
    IbDs,
    get_ocr_features,
    prepare_word_pollys_and_layouts_for_record,
    IBDSConfig,
    get_open_fn,
    get_image_features,
    _DESCRIPTION,
    ExtractionMode,
    ImageProcessingModes,
    get_image_processor,
)
from instabase.dataset_utils.shared_types import ExtractionFieldDict

TEXT_FIELD_TYPE = "Text"
TABLE_FIELD_TYPE = "Table"

# Define some helper classes for easier typing
@dataclass
class ExtractionItem:
    dataset_id: str
    ann_item: "AnnotationItem"
    class_id: Optional[str]
    label2ann_label_id: Dict
    table_label_to_id: Dict


class LabelEntity(TypedDict):
    name: str
    order_id: int
    text: str
    char_spans: List[Span]
    token_spans: List[Span]
    token_label_id: int


def get_extraction_labels(
    ocr_features: Dict,
    annotations: Optional[List[ExtractionFieldDict]] = None,
    label2id: Optional[Dict[str, int]] = None,
    label2ann_label_id: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Process annotations to the format expected by arrow writer
    :param ocr_features: Dict of features obtained from ocr
    :param annotations: List of ExtractionFieldDict for give record
    :param label2ann_label_id: mapping from entity name to entity id used in annotation object
    :param label2id: dictionary which contain mapping from the entity name to class_id
    :return: Dict containing label entities and token_label_ids
    """

    word_line_idx, word_in_line_idx = ocr_features["word_line_idx"], ocr_features["word_in_line_idx"]
    position_map = {
        (line_idx, wrd_idx): gidx for gidx, (line_idx, wrd_idx) in enumerate(zip(word_line_idx, word_in_line_idx))
    }

    token_label_ids = np.zeros((len(word_line_idx)), dtype=np.int64)
    entities = []

    for label_name, lab_id in label2id.items():
        if lab_id == 0 and label_name == "O":
            continue

        ann_label_id = label2ann_label_id[label_name]

        entity_annotations = [ann for ann in annotations if ann["id"] == ann_label_id]
        if len(entity_annotations) > 1:
            raise ValueError("More than one ExtractionFieldDict for given entity")

        if len(entity_annotations) == 0:
            # add empty entity
            entity: LabelEntity = LabelEntity(
                name=label_name,
                order_id=0,
                text="",
                char_spans=[],
                token_spans=[],
                token_label_id=lab_id,
            )

        else:
            extraction_field_ann = entity_annotations[0]["annotations"]
            if len(extraction_field_ann) > 1:
                # raise error as multi item annotations need to be supported by modelling part
                raise ValueError("Mulitple item annotations are not supported yet")

            # Add empty entity if no annotations
            entity: LabelEntity = LabelEntity(
                name=label_name,
                order_id=0,
                text="",
                char_spans=[],
                token_spans=[],
                token_label_id=lab_id,
            )

            for order_id, extraction_ann_dict in enumerate(extraction_field_ann):
                value = extraction_ann_dict["value"]
                label_indexes = []
                for idx_word in extraction_ann_dict.get("words", []):
                    # get global position
                    word_id_global = position_map.get((idx_word["line_index"], idx_word["word_index"]))
                    if word_id_global is None:
                        raise ValueError(f"Cannot find indexed_word in the document - {idx_word}")

                    label_indexes.append(word_id_global)

                label_indexes.sort()
                label_groups = [list(group) for group in consecutive_groups(label_indexes)]
                # create spans for groups, span will be created by getting id for first and last word in the group
                label_token_spans = [[group[0], group[-1] + 1] for group in label_groups]

                for span in label_token_spans:
                    token_label_ids[span[0] : span[1]] = lab_id

                entity: LabelEntity = LabelEntity(
                    name=label_name,
                    order_id=0,
                    text=value,
                    char_spans=[],
                    token_spans=label_token_spans,
                    token_label_id=lab_id,
                )
        entities.append(entity)

    return {"entities": entities, "token_label_ids": token_label_ids}


def get_empty_extraction_labels(
    ocr_features: Dict,
    annotations: Optional[List[ExtractionFieldDict]] = None,
    label2id: Optional[Dict[str, int]] = None,
    label2ann_label_id: Optional[Dict[str, str]] = None,
) -> Dict:
    word_line_idx = ocr_features["word_line_idx"]
    token_label_ids = np.zeros((len(word_line_idx)), dtype=np.int64)
    entities = []
    if label2id is None:
        return {"entities": entities, "token_label_ids": token_label_ids}
    for label_name, lab_id in label2id.items():
        if lab_id == 0 and label_name == "O":
            continue

        ann_label_id = label2ann_label_id[label_name]
        entity: LabelEntity = LabelEntity(
            name=label_name,
            order_id=0,
            text="",
            char_spans=[],
            token_spans=[],
            token_label_id=lab_id,
        )
        entities.append(entity)
    return {"entities": entities, "token_label_ids": token_label_ids}


def get_bb_from_backend(bb, scale_factor):
    bbox = [bb["top_x"], bb["top_y"], bb["bottom_x"], bb["bottom_y"]]
    norm_bb = [int(coord * scale_factor) for coord in bbox]
    return norm_bb


def get_table_labels(
    ocr_features: Dict,
    annotations: Optional[List[ExtractionFieldDict]] = None,
    label2id: Optional[Dict[str, int]] = None,
    label2ann_label_id: Optional[Dict[str, str]] = None,
) -> Dict:

    ann_label_id2label = {v: k for k, v in label2ann_label_id.items()}

    page_bboxes = ocr_features["page_original_bboxes"]

    out_tables = []
    tab_no = 0
    for ann in annotations:
        if not ann["id"] in ann_label_id2label:
            # it's not a table annotation
            continue
        table_label = ann_label_id2label[ann["id"]]
        table_id = label2id[table_label]

        for tab in ann["table_annotations"]:
            out_tab = get_table(tab, table_id, tab_no, page_bboxes)
            out_tables.append(out_tab)
            tab_no += 1

    return {"tables": out_tables}


def get_table(tab, table_id, tab_no, page_bboxes):
    if len(tab["bboxes"]) > 1:
        raise ValueError("mulitple bboxes tables are not supported")

    t_bb = tab["bboxes"][0]

    page_idx = tab["page_indexes"][0]
    scale_factor = 1000 / max(page_bboxes[page_idx])

    out_tab = {
        "table_idx": tab_no,
        "table_label_id": table_id,
        "table_bboxes": [get_bb_from_backend(t_bb, scale_factor)],
        "start_page_index": page_idx,
        "end_page_index": page_idx,
        "rows": [],
        "columns": [],
        "cells": [],
    }
    # process rows
    for row in tab["rows"]:
        out_row = {
            "page_idx": row["page_index"],
            "bbox": get_bb_from_backend(row["bbox"], scale_factor),
            "is_header": row["is_header"],
            "label_id": 0,
        }

        out_tab["rows"].append(out_row)
    # process cols
    for col in tab["cols"]:
        if len(col["page_indexes"]) > 1:
            raise ValueError("multiple page tables are not supported")
        out_col = {
            "page_idx": col["page_indexes"][0],
            "bbox": get_bb_from_backend(col["bboxes"][0], scale_factor),
            "is_header": col["is_header"],
            "label_id": 0,
        }

        out_tab["columns"].append(out_col)
    # process cells
    for cell in tab["cells"]:
        if cell["row_start_index"] != cell["row_end_index"] or cell["col_start_index"] != cell["col_end_index"]:
            raise ValueError("merged cells are not supported")
        out_cell = {
            "page_idx": cell["page_index"],
            "bbox": get_bb_from_backend(cell["bbox"], scale_factor),
            "row_start_index": cell["row_start_index"],
            "row_end_index": cell["row_end_index"],
            "col_start_index": cell["col_start_index"],
            "col_end_index": cell["col_end_index"],
            "is_row_header": False,
            "is_column_header": False,
        }

        out_tab["cells"].append(out_cell)
    return out_tab


def get_empty_table_labels(
    ocr_features: Dict,
    annotations: Optional[List[ExtractionFieldDict]] = None,
    label2id: Optional[Dict[str, int]] = None,
    label2ann_label_id: Optional[Dict[str, str]] = None,
) -> Dict:
    return {"tables": []}


def get_extraction_split(anno: Optional[Dict], extraction_mode: ExtractionMode = ExtractionMode.TEXT):
    """
    :param anno: annotation dictionary
    :param extraction_mode: current extraction mode
    :return: Tuple of boolean indictating whether file is marked as test file and split information
    """
    if anno is None or len(anno) == 0:
        return False, "predict"
    exist_any_text_annotations = any(
        [len(ann.get("words", [])) > 0 for fld in anno.get("fields", []) for ann in fld.get("annotations", [])]
    )
    exist_any_table_annotations = any(
        [len(ann.get("bboxes", [])) > 0 for fld in anno.get("fields", []) for ann in fld.get("table_annotations", [])]
    )
    if extraction_mode == ExtractionMode.TEXT:
        exist_any_annotations = exist_any_text_annotations
    elif extraction_mode == ExtractionMode.TABLE:
        exist_any_annotations = exist_any_table_annotations
    else:
        raise ValueError(f"Unknown extraction mode: {extraction_mode}")
    if not exist_any_annotations:
        return False, "predict"
    if anno["is_test_file"]:
        return True, "test"
    return False, "train"


class IbExtraction(IbDs):
    """
    Instabase internal dataset format, creation of dataset can be done by passing list of datasets
    """

    CONFIG_NAME = "ib_extraction"

    BUILDER_CONFIGS = [
        IBDSConfig(
            name="ib_extraction",
            version=datasets.Version("1.0.0"),
            description="Instabase Format Datasets",
            image_size=224,
        ),
        IBDSConfig(
            name="ib_table_extraction",
            version=datasets.Version("1.0.0"),
            description="Instabase Format Datasets",
            extraction_mode=ExtractionMode.TABLE,
            processing_mode=ImageProcessingModes.DETR,
            norm_bboxes_to_max=True,
            image_size=1000,
        ),
    ]

    def get_class_id(self, dataset_classes):
        matching_class_ids = [
            class_id
            for class_id, class_def in dataset_classes.items()
            if class_def["name"] == self.config.extraction_class_name
        ]
        if len(matching_class_ids) == 0:
            raise ValueError("extraction_class_name not found in dataset")
        return matching_class_ids[0]

    @classmethod
    def create_image_processor(cls, config: IBDSConfig):
        if not config.use_image:
            return None
        return get_image_processor(config)

    def _info(self):

        data_files = self.config.data_files
        assert isinstance(data_files, dict), "data_files argument should be a dict for this dataset"
        if "train" in data_files:
            features = self.get_train_dataset_features()
            label2id = features["token_label_ids"].feature._str2int
            table_label2id = features["tables"].feature["table_label_id"]._str2int
            self.config.id2label = {v: k for k, v in label2id.items()}
            self.config.table_id2label = {v: k for k, v in table_label2id.items()}
        elif "test" in data_files:
            features = self.get_inference_dataset_features(self.config)
        else:
            raise ValueError("data_file argument should be either in train or test mode")

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
        )

    def get_train_dataset_features(self) -> datasets.Features:
        data_files = self.config.data_files
        self.datasets_list = load_datasets(data_files["train"], self.config.ibsdk)
        dataset_classes = self.datasets_list[0].metadata["classes_spec"]["classes"]
        class_id = self.get_class_id(dataset_classes)
        schema = dataset_classes[class_id]["schema"]
        classes = ["O"] + [lab["name"] for lab in schema if lab.get("type", TEXT_FIELD_TYPE) == TEXT_FIELD_TYPE]
        # need to assign O label for tables as well
        table_classes = ["O"] + [lab["name"] for lab in schema if lab.get("type", TEXT_FIELD_TYPE) == TABLE_FIELD_TYPE]
        return self.create_dataset_features(self.config, classes, table_classes)

    @classmethod
    def get_inference_dataset_features(cls, config: IBDSConfig) -> datasets.Features:
        if config.extraction_mode == ExtractionMode.TEXT:
            assert config.id2label is not None, "Need to pass directly infromation about labels for the inference"
            classes = [config.id2label[i] for i in range(len(config.id2label))]
            if classes[0] != "O":
                raise logging.error(f"loaded classes does not have required format. No O class: {classes}")
            return cls.create_dataset_features(config, classes, [])

        elif config.extraction_mode == ExtractionMode.TABLE:
            assert config.table_id2label is not None, "Need to pass directly infromation about labels for the inference"
            table_classes = [config.table_id2label[i] for i in range(len(config.table_id2label))]
            if table_classes[0] != "O":
                raise logging.error(f"loaded classes does not have required format. No O class: {table_classes}")
            return cls.create_dataset_features(config, [], table_classes)

    @staticmethod
    def create_dataset_features(config, classes, table_classes):
        if len(classes) == 0:
            classes = ["O"]
        if len(table_classes) == 0:
            table_classes = ["O"]
        ds_features = get_common_feature_schema(config=config)
        ds_features["token_label_ids"] = datasets.Sequence(datasets.features.ClassLabel(names=classes))
        ds_features["entities"] = datasets.Sequence(
            {
                "name": datasets.Value("string"),
                "order_id": datasets.Value("int64"),  # not supported yet, annotation app need to implement it
                "text": datasets.Value("string"),
                "char_spans": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=2)),
                "token_spans": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=2)),
                "token_label_id": datasets.Value("int64"),
            }
        )
        ds_features["tables"] = datasets.Sequence(
            {
                "table_idx": datasets.Value("int32"),
                "table_label_id": datasets.features.ClassLabel(names=table_classes),
                "start_page_index": datasets.Value("int32"),
                "end_page_index": datasets.Value("int32"),
                "table_bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=4)),
                "rows": datasets.Sequence(
                    {
                        "page_idx": datasets.Value("int32"),
                        "bbox": datasets.Sequence(datasets.Value("int32"), length=4),
                        "is_header": datasets.Value("bool"),
                        "label_id": datasets.Value("int32"),
                    }
                ),
                "columns": datasets.Sequence(
                    {
                        "page_idx": datasets.Value("int32"),
                        "bbox": datasets.Sequence(datasets.Value("int32"), length=4),
                        "is_header": datasets.Value("bool"),
                        "label_id": datasets.Value("int32"),
                    }
                ),
                "cells": datasets.Sequence(
                    {
                        "row_start_index": datasets.Value("int32"),
                        "row_end_index": datasets.Value("int32"),
                        "col_start_index": datasets.Value("int32"),
                        "col_end_index": datasets.Value("int32"),
                        "page_idx": datasets.Value("int32"),
                        "bbox": datasets.Sequence(datasets.Value("int32"), length=4),
                        "is_row_header": datasets.Value("bool"),
                        "is_column_header": datasets.Value("bool"),
                    }
                ),
            }
        )
        return datasets.Features(ds_features)

    def _get_annotation_generator(self, datasets_list: List["DatasetSDK"]) -> Iterable[ExtractionItem]:
        logging.info(f"Reading from Instabase datasets, extracting class name: {self.extraction_class_name}")

        for dataset in datasets_list:
            # first determine the class name's corresponding id.
            dataset_classes = dataset.metadata["classes_spec"]["classes"]
            class_id = self.get_class_id(dataset_classes)
            dataset_id = dataset.metadata["id"]
            # then get all records with this class id and their annotations
            class_schema = dataset_classes[class_id]["schema"]
            label2ann_label_id = {
                field["name"]: field["id"] for field in class_schema if field["type"] == TEXT_FIELD_TYPE
            }
            table_label_to_id = {
                field["name"]: field["id"] for field in class_schema if field["type"] == TABLE_FIELD_TYPE
            }

            for record_anno in dataset.iterator_over_annotations():
                item = ExtractionItem(
                    ann_item=record_anno,
                    dataset_id=dataset_id,
                    class_id=class_id,
                    label2ann_label_id=label2ann_label_id,
                    table_label_to_id=table_label_to_id,
                )
                yield item

    @classmethod
    def _get_annotation_from_model_service(self, records: List[Tuple], config: IBDSConfig) -> Iterable[ExtractionItem]:
        # get similar format to the one defined by dataset SDK
        for prediction_item in records:
            label2ann_label_id = {lab: None for lab in config.id2label.values()} if config.id2label is not None else {}
            table_label_to_id = (
                {lab: None for lab in config.table_id2label.values()} if config.table_id2label is not None else {}
            )
            item = ExtractionItem(
                ann_item=prediction_item,
                dataset_id="model-inference",
                class_id=None,
                label2ann_label_id=label2ann_label_id,
                table_label_to_id=table_label_to_id,
            )
            yield item

    @classmethod
    def process_item(cls, item: ExtractionItem, config: IBDSConfig, image_processor: ImageProcessor) -> Dict:
        full_path, record_index, record, anno = item.ann_item

        anno = {} if anno is None else anno
        anno_fields = anno.get("fields", [])
        annotated_class_id = anno.get("annotated_class_id")
        if annotated_class_id is not None and annotated_class_id != item.class_id:
            logging.info("Skipping this document because it is marked as a different class")
            return None

        err = assert_valid_record(record)
        if err is not None:
            logging.error(f"Skipping this document because the IBDOC has corrupt OCR words: {err}")
            return None

        logging.info(f"Processing record_idx={record_index}\t path={full_path}")
        words, layouts = prepare_word_pollys_and_layouts_for_record(record)
        doc_id = f"{item.dataset_id}-{os.path.basename(full_path)}-{record_index}.json"

        ocr_features = get_ocr_features(words, layouts, doc_id, config.norm_bboxes_to_max, config.bbox_scale_factor)
        extraction_label_getter = (
            get_extraction_labels if config.extraction_mode == ExtractionMode.TEXT else get_empty_extraction_labels
        )
        label_features = extraction_label_getter(ocr_features, anno_fields, config.label2id, item.label2ann_label_id)

        is_test_file, split = get_extraction_split(anno, config.extraction_mode)
        open_fn = get_open_fn(config.ibsdk)
        image_features = (
            get_image_features(ocr_features, layouts, full_path, open_fn, image_processor)
            if image_processor is not None
            else {}
        )

        # TODO (BT): Maybe we should leave all the labels present, it works ok
        table_label_getter = (
            get_table_labels if config.extraction_mode == ExtractionMode.TABLE else get_empty_table_labels
        )
        table_features = table_label_getter(ocr_features, anno_fields, config.table_label2id, item.table_label_to_id)
        return {
            **ocr_features,
            **label_features,
            **image_features,
            "is_test_file": is_test_file,
            "split": split,
            "id": doc_id,
            **table_features,
        }
