import itertools
import logging
import os
from dataclasses import dataclass
from typing import Tuple, List, Dict, Union, Optional, Sequence, Mapping, Iterable, Any

import datasets
from datasets import DatasetInfo

from ibformers.data.utils import ImageProcessor
from ibformers.datasets.ib_common.ib_common import (
    assert_valid_record,
    get_common_feature_schema,
    load_datasets,
    IbDs,
    IBDSConfig,
    prepare_word_pollys_and_layouts_for_record,
    get_ocr_features,
    get_open_fn,
    get_image_features,
)


@dataclass
class ClassItem:
    dataset_id: str
    ann_item: "AnnotationItem"
    class_id2class_label: Dict


def get_classification_split(anno: Optional[Dict]):
    """
    :param anno: annotation dictionary
    :return: Tuple of boolean indictating whether file is marked as test file and split information
    """
    if anno is None:
        return False, "predict"
    if "is_test_file" not in anno:
        return False, "predict"
    if anno["is_test_file"]:
        return True, "test"

    return False, "train"


def get_annotation_features(anno, class_id2label):
    """
    process annotation_item feeded by Dataset SDK into dictionary yielded directly to Arrow writer
    :param anno: AnnotationItem object return by Dataset SDK or created from model service request
    :param dataset_id: id of the dataset of the processed record
    :param class_label2id:
    :return:
    """

    if anno is None:
        anno = {}
    annotated_class_id = anno.get("annotated_class_id")

    is_test_file, split = get_classification_split(anno)

    return {
        "class_label": class_id2label[annotated_class_id] if annotated_class_id is not None else 0,
        "split": split,
        "is_test_file": is_test_file,
    }


class IbClassificationDs(IbDs):
    """
    Instabase internal dataset format, creation of dataset can be done by passing list of datasets
    """

    CONFIG_NAME = "ib_classification"

    BUILDER_CONFIGS = [
        IBDSConfig(
            name=CONFIG_NAME,
            version=datasets.Version("1.0.0"),
            description="Instabase Format Datasets",
        ),
    ]

    def get_train_dataset_features(self) -> datasets.Features:
        data_files: Union[str, Sequence, Mapping] = self.config.data_files
        self.datasets_list = load_datasets(data_files["train"], self.config.ibsdk)
        dataset_classes = dict(
            itertools.chain(*(dataset.metadata["classes_spec"]["classes"].items() for dataset in self.datasets_list))
        )
        classes = list(set([data_class["name"] for data_class in dataset_classes.values()]))
        return self.create_dataset_features(self.config, classes)

    @classmethod
    def get_inference_dataset_features(cls, config: IBDSConfig) -> datasets.Features:
        assert config.id2label is not None, "Need to pass directly infromation about labels for the inference"
        classes = [config.id2label[i] for i in range(len(config.id2label))]
        return cls.create_dataset_features(config, classes)

    @staticmethod
    def create_dataset_features(config, classes):
        ds_features = get_common_feature_schema(config=config)
        ds_features["class_label"] = datasets.features.ClassLabel(names=classes)
        return datasets.Features(ds_features)

    def _get_annotation_generator(self, datasets_list: List["DatasetSDK"]) -> Iterable[ClassItem]:
        logging.info(f"Reading from Instabase datasets")

        for dataset in datasets_list:
            dataset_classes = dataset.metadata["classes_spec"]["classes"]
            class_id2class_label = {
                data_class_id: data_class["name"] for data_class_id, data_class in dataset_classes.items()
            }
            dataset_id = dataset.metadata["id"]
            for record_anno in dataset.iterator_over_annotations():
                item = ClassItem(ann_item=record_anno, dataset_id=dataset_id, class_id2class_label=class_id2class_label)
                yield item

    @classmethod
    def _get_annotation_from_model_service(self, records: List[Tuple], config: IBDSConfig) -> Iterable[ClassItem]:
        # get similar format to the one defined by dataset SDK
        for prediction_item in records:
            item = ClassItem(
                ann_item=prediction_item, dataset_id="model-inference", class_id2class_label=config.id2label
            )
            yield item

    @classmethod
    def process_item(cls, item: ClassItem, config: IBDSConfig, image_processor: ImageProcessor) -> Dict:
        full_path, record_index, record, anno = item.ann_item

        logging.info(f"Processing record_idx={record_index}\t path={full_path}")

        err = assert_valid_record(record)
        if err is not None:
            logging.error(f"Skipping this document because the IBDOC has corrupt OCR words: {err}")
            return None

        words, layouts = prepare_word_pollys_and_layouts_for_record(record)
        doc_id = f"{item.dataset_id}-{os.path.basename(full_path)}-{record_index}.json"
        ocr_features = get_ocr_features(words, layouts, doc_id)

        anno_features = get_annotation_features(anno, item.class_id2class_label)
        open_fn = get_open_fn(config.ibsdk)
        image_features = (
            get_image_features(ocr_features, layouts, full_path, open_fn, image_processor) if config.use_image else {}
        )

        return {**ocr_features, **anno_features, **image_features, "id": doc_id}
