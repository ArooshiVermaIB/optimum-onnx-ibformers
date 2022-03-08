import itertools
import logging
import os
from dataclasses import dataclass

from typing import Dict, List, Iterable

import datasets

from ibformers.datasets.ib_common import (
    assert_valid_record,
    load_datasets,
    IBDSConfig,
    get_common_feature_schema,
    IbDs,
    prepare_word_pollys_and_layouts_for_record,
    get_ocr_features,
    get_open_fn,
    get_image_features,
)
from instabase.ocr.client.libs.ibocr import ParsedIBOCR

_DESCRIPTION = """\
Internal Instabase Dataset format organized into set of IbDoc files for splitting.
"""


@dataclass
class SplitClassItem:
    dataset_id: str
    pibocr: ParsedIBOCR
    class_id2class_label: Dict


class IbSplitClass(IbDs):
    """
    Instabase internal dataset format, creation of dataset can be done by passing list of datasets
    """

    CONFIG_NAME = "ib_split_class"

    BUILDER_CONFIGS = [
        IBDSConfig(
            name=CONFIG_NAME,
            version=datasets.Version("1.0.0"),
            description="Instabase Format Datasets",
        ),
    ]

    def _info(self):
        data_files = self.config.data_files
        assert isinstance(data_files, dict), "data_files argument should be a dict for this dataset"
        if "train" in data_files:
            self.datasets_list = load_datasets(data_files["train"], self.config.ibsdk)
            dataset_classes = dict(
                itertools.chain(
                    *(dataset.metadata["classes_spec"]["classes"].items() for dataset in self.datasets_list)
                )
            )
            classes = list(set([class_def["name"] for _, class_def in dataset_classes.items()]))
        elif "test" in data_files:
            # inference input is a list of parsedibocr files
            assert self.config.id2label is not None, "Need to pass directly infromation about labels for the inference"
            classes = [self.config.id2label[i] for i in range(len(self.config.id2label))]
        else:
            raise ValueError("data_file argument should be either in train or test mode")

        ds_features = get_common_feature_schema(use_image=self.config.use_image)
        ds_features["record_page_ranges"] = datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=2))
        ds_features["class_label"] = datasets.Sequence(datasets.features.ClassLabel(names=classes))

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            features=datasets.Features(ds_features),
            supervised_keys=None,
        )

    def _get_annotation_generator(self, datasets_list) -> Iterable[SplitClassItem]:
        logging.info(f"Reading from Instabase datasets")
        for dataset in datasets_list:
            # TODO: confirm with Vineeth if class_id2name could be on dataset level
            dataset_classes = dataset.metadata["classes_spec"]["classes"]
            class_id2name = {class_id: class_def["name"] for class_id, class_def in dataset_classes.items()}
            dataset_id = dataset.metadata["id"]
            for pibocr in dataset.iterator_over_ibdocs():
                item = SplitClassItem(pibocr=pibocr, dataset_id=dataset_id, class_id2class_label=class_id2name)
                yield item

    def _get_annotation_from_model_service(self, pibocrs: List[ParsedIBOCR]) -> Iterable[SplitClassItem]:
        for pibocr in pibocrs:
            class_id2name = {lab: None for lab in self.config.id2label.values()}
            item = SplitClassItem(pibocr=pibocr, dataset_id="model_service", class_id2class_label=class_id2name)
            yield item

    def process_item(self, item: SplitClassItem) -> Dict:
        class_name2int = self.info.features["class_label"].feature._str2int
        words = []
        page_ranges = []
        class_labels = []
        is_test = False
        is_predict = False
        # iterate through all records in the ibdoc
        for record_idx, record in enumerate(item.pibocr.get_ibocr_records()):

            err = assert_valid_record(record)
            if err is not None:
                logging.error(f"Skipping this document because the IBDOC has corrupt OCR words: {err}")
                continue

            rwords, rlayouts = prepare_word_pollys_and_layouts_for_record(record)
            words.extend(rwords)

            page_nums = record.get_page_numbers()
            page_ranges.append([min(page_nums), max(page_nums)])
            anno = record.get_doc_extraction_annotations()

            if anno is None:
                class_labels.append(0)
                is_predict = True
            else:
                class_id = anno["annotated_class_id"]
                is_test = (is_test or anno["is_test_file"]) and not is_predict
                class_label = class_name2int[item.class_id2class_label[class_id]]
                class_labels.append(class_label)
        # assign layouts of any record
        layouts = rlayouts
        doc_path = record.get_absolute_ibocr_path()
        doc_id = f"{item.dataset_id}-{os.path.basename(doc_path)}"

        if is_test:
            split_type = "test"
        elif is_predict:
            split_type = "predict"
        else:
            split_type = "train"

        ocr_features = get_ocr_features(words, layouts, doc_id)
        open_fn = get_open_fn(self.config.ibsdk)
        image_features = (
            get_image_features(ocr_features, layouts, doc_path, open_fn, self.image_processor)
            if self.config.use_image
            else {}
        )

        return {
            **ocr_features,
            **image_features,
            "record_page_ranges": page_ranges,
            "class_label": class_labels,
            "is_test_file": is_test,
            "split": split_type,
            "id": doc_id,
        }
