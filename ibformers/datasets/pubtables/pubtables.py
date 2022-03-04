import hashlib
import json
import logging
import multiprocessing
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Iterable, Sequence

import datasets
import numpy as np
from datasets import BuilderConfig, DatasetInfo, DownloadManager

from ibformers.data.utils import ImageProcessor
from ibformers.datasets.docpro_ds.docpro_ds import validate_and_fix_bboxes
from ibformers.datasets.table_utils import (
    CellAnnotation,
    TableAnnotation,
    DocTableAnnotations,
    RowAnnotation,
    ColumnAnnotation,
)

logger = logging.getLogger(__name__)

HASH_MODULO = 1000000
DATASET_SPLITS = 0.97, 0.015, 0.015


def pubtables_cell_to_cell_annotation(raw_cell_data: Dict[str, Any]) -> CellAnnotation:
    return CellAnnotation(
        row_ids=raw_cell_data["row_nums"],
        col_ids=raw_cell_data["column_nums"],
        bbox=raw_cell_data["pdf_bbox"],
        page_idx=0,
        is_column_header=raw_cell_data["is_column_header"],
        is_row_header=raw_cell_data["is_projected_row_header"],
    )


def pubtables_anno_to_table_annotation(raw_table_data: Dict[str, Any], current_table_id: int) -> TableAnnotation:
    base_page_bboxes = [raw_table_data["pdf_full_page_bbox"]]
    table_bboxes = [raw_table_data["pdf_table_bbox"]]
    rows = [RowAnnotation(r["pdf_row_bbox"], 0, r["is_column_header"]) for r in raw_table_data["rows"]]
    columns = [ColumnAnnotation(r["pdf_column_bbox"], 0) for r in raw_table_data["columns"]]
    cells = [pubtables_cell_to_cell_annotation(cell_data) for cell_data in raw_table_data["cells"]]
    return TableAnnotation(
        table_idx=current_table_id,
        cells=cells,
        page_span=(0, 1),
        base_page_bboxes=base_page_bboxes,
        table_bboxes=table_bboxes,
        rows=rows,
        columns=columns,
    )


def pubtables_line_to_doc_table_annotations(annotation_line: List[Dict[str, Any]]) -> Iterable[DocTableAnnotations]:
    current_page_no = None
    doc_id = None
    current_table_id = 0
    current_page_table_annotations = []
    annotation_line = sorted(annotation_line, key=lambda x: x["pdf_page_index"])
    for raw_table_data in annotation_line:
        if current_page_no is None:
            current_page_no = raw_table_data["pdf_page_index"]
        if doc_id is None:
            doc_id = raw_table_data["pmc_id"]
        if raw_table_data["pdf_page_index"] != current_page_no:
            yield DocTableAnnotations(
                anno_id=f"{doc_id}_{current_page_no}", table_annotations=current_page_table_annotations
            )
            current_table_id = 0
            current_page_table_annotations = []
        table_annotation = pubtables_anno_to_table_annotation(raw_table_data, current_table_id)
        current_page_table_annotations.append(table_annotation)
        current_table_id += 1
        current_page_no = raw_table_data["pdf_page_index"]

    if doc_id is not None:
        yield DocTableAnnotations(
            anno_id=f"{doc_id}_{current_page_no}", table_annotations=current_page_table_annotations
        )


def read_single_pubtable_anno(anno_path) -> List[DocTableAnnotations]:
    content = json.loads(anno_path.read_text())
    return [doc_table_anno for doc_table_anno in pubtables_line_to_doc_table_annotations(content)]


def read_and_process_pdf_words(pdf_words_path: Path):
    content = json.loads(pdf_words_path.read_text())
    features_dict = {
        "words": [item["text"] for item in content["words"]],
        "word_original_bboxes": np.array([item["bbox"] for item in content["words"]]),
        "page_original_bboxes": [content["page_rect"]],
    }
    total_words = len(features_dict["words"])
    features_dict["page_spans"] = [[0, total_words]]
    features_dict["word_page_nums"] = [0] * total_words
    features_dict["word_line_idx"] = [0] * total_words
    features_dict["word_in_line_idx"] = [0] * total_words
    return features_dict


class PubTablesConfig(BuilderConfig):
    def __init__(self, use_image: bool = False, num_processes: int = 12, limit_files: int = None, **kwargs):
        super().__init__(**kwargs)
        self.use_image = use_image
        self.num_processes = num_processes
        self.limit_files = limit_files


class PubTables(datasets.GeneratorBasedBuilder):
    """
    Dataset for pre-training table extraction models on Pubtables dataset
    """

    BUILDER_CONFIGS = [
        PubTablesConfig(
            name="pubtables",
            version=datasets.Version("1.0.0"),
            description="PubTables dataset",
        ),
    ]

    ANNOTATION_SUBDIR = "PubTables1M-PDF-Annotations-JSON"
    PAGE_WORDS_SUBDIR = "PubTables1M-Page-Words-JSON"
    IMAGE_SUBDIR = "PubTables1M-Detection-PASCAL-VOC/images"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_processor = ImageProcessor(do_resize=True, size=224) if self.config.use_image else None
        self.num_processes = self.config.num_processes
        self.limit_files = self.config.limit_files

    def _info(self) -> DatasetInfo:
        ds_features = {
            "id": datasets.Value("string"),
            "words": datasets.Sequence(datasets.Value("string")),
            "bboxes": datasets.Array2D(shape=(None, 4), dtype="int32"),
            # needed to generate prediction file, after evaluation
            "word_original_bboxes": datasets.Array2D(shape=(None, 4), dtype="float32"),
            "word_page_nums": datasets.Sequence(datasets.Value("int32")),
            "word_line_idx": datasets.Sequence(datasets.Value("int32")),
            "word_in_line_idx": datasets.Sequence(datasets.Value("int32")),
            "page_bboxes": datasets.Array2D(shape=(None, 4), dtype="int32"),
            "page_spans": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=2)),
            "tables": datasets.Sequence(
                {
                    "table_idx": datasets.Value("int32"),
                    "page_span": datasets.Sequence(datasets.Value("int32"), length=2),  # table should be continuous
                    "table_bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=4)),
                    "rows": datasets.Sequence(
                        {
                            "page_idx": datasets.Value("int32"),
                            "bbox": datasets.Sequence(datasets.Value("int32"), length=4),
                            "is_column_header": datasets.Value("bool"),
                        }
                    ),
                    "columns": datasets.Sequence(
                        {
                            "page_idx": datasets.Value("int32"),
                            "bbox": datasets.Sequence(datasets.Value("int32"), length=4),
                        }
                    ),
                    "cells": datasets.Sequence(
                        {
                            "row_ids": datasets.Sequence(datasets.Value("int32")),
                            "col_ids": datasets.Sequence(datasets.Value("int32")),
                            "page_idx": datasets.Value("int32"),
                            "bbox": datasets.Sequence(datasets.Value("int32"), length=4),
                            "is_row_header": datasets.Value("bool"),
                            "is_column_header": datasets.Value("bool"),
                        }
                    ),
                }
            ),
        }
        if self.config.use_image:
            # first dimension is defined as a number of pages in the document
            ds_features["images"] = datasets.Array4D(shape=(None, 3, 224, 224), dtype="uint8")
            ds_features["images_page_nums"] = datasets.Sequence(datasets.Value("int32"))

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description="PubTables-1M implementation",
            features=datasets.Features(ds_features),
            supervised_keys=None,
        )

    @staticmethod
    def get_split(file_path: Path):
        hex_score = hashlib.md5(str(file_path.stem).encode()).hexdigest()
        int_score = int(hex_score, 16)
        p_score = int_score % HASH_MODULO / HASH_MODULO
        if p_score < DATASET_SPLITS[0]:
            return datasets.Split.TRAIN
        elif p_score < DATASET_SPLITS[0] + DATASET_SPLITS[1]:
            return datasets.Split.VALIDATION
        else:
            return datasets.Split.TEST

    def _get_pdf_anno_paths(self, dataset_dir: Path) -> List[Path]:
        return list((dataset_dir / self.ANNOTATION_SUBDIR).glob("*.json"))[: self.limit_files]

    def _get_word_path(self, pdf_anno_path: Path, anno_id: str) -> Path:
        word_anno_dir = pdf_anno_path.parent.parent / self.PAGE_WORDS_SUBDIR
        return word_anno_dir / f"{anno_id}_words.json"

    def _get_image_path(self, pdf_anno_path: Path, anno_id: str) -> Path:
        image_dir = pdf_anno_path.parent.parent / self.IMAGE_SUBDIR
        return image_dir / f"{anno_id}.jpg"

    def _split_generators(self, dl_manager: DownloadManager):
        if "train" not in self.config.data_files:
            raise ValueError("Please provide a path to the index as train file")
        dataset_dir = Path(self.config.data_files["train"])
        anno_files = self._get_pdf_anno_paths(dataset_dir)
        splits = [self.get_split(path) for path in anno_files]
        counts = Counter(splits)
        logger.info(f"Dataset counts: {counts.most_common()}")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "index_content": [
                        paths for paths, split in zip(anno_files, splits) if split == datasets.Split.TRAIN
                    ]
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "index_content": [
                        paths for paths, split in zip(anno_files, splits) if split == datasets.Split.VALIDATION
                    ]
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "index_content": [paths for paths, split in zip(anno_files, splits) if split == datasets.Split.TEST]
                },
            ),
        ]

    def _scale_bbox(self, bbox: Sequence[int], scale_factor: int) -> Sequence[int]:
        return [int(coord * 1000 / scale_factor) for coord in bbox]

    def _prepare_feature_for_layoutlm(self, feature: Dict[str, Any]):
        feature["bboxes"] = np.array(feature["word_original_bboxes"])
        feature["page_bboxes"] = np.array(feature.pop("page_original_bboxes"))
        word_pages_arr = np.array(feature["word_page_nums"])
        size_per_token = np.take(feature["page_bboxes"][:, 2:], word_pages_arr, axis=0)
        fix_bbox_arr = validate_and_fix_bboxes(
            feature["bboxes"], size_per_token, word_pages_arr, feature["page_bboxes"], feature["id"]
        )
        norm_bboxes = fix_bbox_arr * 1000 / size_per_token[:, 0:1]
        feature["bboxes"] = norm_bboxes

        for table in feature["tables"]:
            table["table_bboxes"] = [
                self._scale_bbox(box, feature["page_bboxes"][page_index][2])
                for box, page_index in zip(table["table_bboxes"], range(table["page_span"][0], table["page_span"][1]))
            ]
            for structure_name in ["cells", "rows", "columns"]:
                for structure in table[structure_name]:
                    structure["bbox"] = self._scale_bbox(
                        structure["bbox"], feature["page_bboxes"][structure["page_idx"]][2]
                    )

        # TODO: fix the bbox rescaling to do it all in the pipeline
        feature["page_bboxes"] = feature["page_bboxes"] * 1000 / feature["page_bboxes"][:, 2:3]
        return feature

    def _load_single_anno(self, file_path: Path):
        raw_annos = read_single_pubtable_anno(file_path)
        feature_dicts = []
        for anno in raw_annos:
            features_dict = anno.to_dataset_features_part()
            pdf_words_path = self._get_word_path(file_path, anno.anno_id)
            pdf_words_features = read_and_process_pdf_words(pdf_words_path)
            features_dict.update(**pdf_words_features)
            features_dict = self._prepare_feature_for_layoutlm(features_dict)
            feature_dicts.append(features_dict)
        return feature_dicts

    def _generate_examples(self, index_content: List[Path]):
        logger.info(f"Generating {len(index_content)} examples")

        with multiprocessing.Pool(self.num_processes) as pool:
            for doc_list in pool.imap(self._load_single_anno, index_content):
                for doc_dict in doc_list:
                    yield doc_dict["id"], doc_dict
