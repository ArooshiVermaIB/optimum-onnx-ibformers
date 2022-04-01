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
)
from instabase.dataset_utils.shared_types import ExtractionFieldDict


# Define some helper classes for easier typing
@dataclass
class ExtractionItem:
    dataset_id: str
    ann_item: "AnnotationItem"
    class_id: Optional[str]
    label2ann_label_id: Dict


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


def get_extraction_split(anno: Optional[Dict]):
    """
    :param anno: annotation dictionary
    :return: Tuple of boolean indictating whether file is marked as test file and split information
    """
    if anno is None or len(anno) == 0:
        return False, "predict"
    exist_any_annotations = any(
        [len(ann.get("words", [])) > 0 for fld in anno.get("fields", []) for ann in fld["annotations"]]
    )
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
            name=CONFIG_NAME,
            version=datasets.Version("1.0.0"),
            description="Instabase Format Datasets",
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

    def _info(self):

        data_files = self.config.data_files
        assert isinstance(data_files, dict), "data_files argument should be a dict for this dataset"
        if "train" in data_files:
            features = self.get_train_dataset_features()
            label2id = features["token_label_ids"].feature._str2int
            self.config.id2label = {v: k for k, v in label2id.items()}
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
        classes = ["O"] + [lab["name"] for lab in schema]
        return self.create_dataset_features(self.config, classes)

    @classmethod
    def get_inference_dataset_features(cls, config: IBDSConfig) -> datasets.Features:
        assert config.id2label is not None, "Need to pass directly infromation about labels for the inference"
        classes = [config.id2label[i] for i in range(len(config.id2label))]
        if classes[0] != "O":
            raise logging.error(f"loaded classes does not have required format. No O class: {classes}")
        return cls.create_dataset_features(config, classes)

    @staticmethod
    def create_dataset_features(config, classes):
        ds_features = get_common_feature_schema(use_image=config.use_image)
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
            label2ann_label_id = {field["name"]: field["id"] for field in class_schema}

            for record_anno in dataset.iterator_over_annotations():
                item = ExtractionItem(
                    ann_item=record_anno,
                    dataset_id=dataset_id,
                    class_id=class_id,
                    label2ann_label_id=label2ann_label_id,
                )
                yield item

    @classmethod
    def _get_annotation_from_model_service(self, records: List[Tuple], config: IBDSConfig) -> Iterable[ExtractionItem]:
        # get similar format to the one defined by dataset SDK
        for prediction_item in records:
            label2ann_label_id = {lab: None for lab in config.id2label.values()}
            item = ExtractionItem(
                ann_item=prediction_item,
                dataset_id="model-inference",
                class_id=None,
                label2ann_label_id=label2ann_label_id,
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

        ocr_features = get_ocr_features(words, layouts, doc_id)
        label_features = get_extraction_labels(ocr_features, anno_fields, config.label2id, item.label2ann_label_id)
        is_test_file, split = get_extraction_split(anno)
        open_fn = get_open_fn(config.ibsdk)
        image_features = (
            get_image_features(ocr_features, layouts, full_path, open_fn, image_processor)
            if image_processor is not None
            else {}
        )

        return {
            **ocr_features,
            **label_features,
            **image_features,
            "is_test_file": is_test_file,
            "split": split,
            "id": doc_id,
        }
