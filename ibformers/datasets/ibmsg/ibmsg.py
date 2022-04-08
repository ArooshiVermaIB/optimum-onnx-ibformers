import hashlib
import logging
import multiprocessing
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import datasets
import numpy as np
from datasets import BuilderConfig, DatasetInfo, DownloadManager

from ibformers.data.utils import ImageProcessor
from ibformers.datasets.ib_common.ib_common import validate_and_fix_bboxes, assert_valid_record
from instabase.ocr.client.libs.ibocr import ParsedIBOCRBuilder, IBOCRRecord

HASH_MODULO = 1000000
DATASET_SPLITS = 0.9, 0.07, 0.03


logger = logging.getLogger(__name__)


class IbmsgConfig(BuilderConfig):
    def __init__(self, use_image: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.use_image = use_image


class DocLoadingException(BaseException):
    pass


class Ibmsg(datasets.GeneratorBasedBuilder):
    """
    Dataset for unsupervised training over set of ibmsg files.
    """

    BUILDER_CONFIGS = [
        IbmsgConfig(
            name="ibmsg_ds",
            version=datasets.Version("1.0.0"),
            description="IBMsg dataset",
        ),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_processor = ImageProcessor(do_resize=True, size=224) if self.config.use_image else None

    def _info(self) -> DatasetInfo:
        ds_features = {
            "id": datasets.Value("string"),
            # to distiguish test files marked in the doc pro app from all files for which predictions are needed
            # TODO: remove this column once Dataset SDK will allow for test files iterators
            "words": datasets.Sequence(datasets.Value("string")),
            "bboxes": datasets.Array2D(shape=(None, 4), dtype="int32"),
            # needed to generate prediction file, after evaluation
            "word_original_bboxes": datasets.Array2D(shape=(None, 4), dtype="float32"),
            "word_page_nums": datasets.Sequence(datasets.Value("int32")),
            "word_line_idx": datasets.Sequence(datasets.Value("int32")),
            "word_in_line_idx": datasets.Sequence(datasets.Value("int32")),
            "page_bboxes": datasets.Array2D(shape=(None, 4), dtype="int32"),
            "page_spans": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=2)),
        }

        if self.config.use_image:
            # first dimension is defined as a number of pages in the document
            ds_features["images"] = datasets.Array4D(shape=(None, 3, 224, 224), dtype="uint8")
            ds_features["images_page_nums"] = datasets.Sequence(datasets.Value("int32"))
        return DatasetInfo(
            # This is the description that will appear on the datasets page.
            description="IBMsg dataset",
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

    def _split_generators(self, dl_manager: DownloadManager):
        if "train" not in self.config.data_files:
            raise ValueError("Please provide a path to the index as train file")
        index_path = Path(self.config.data_files["train"])
        index_content = self._load_index(index_path)
        splits = [self.get_split(path) for _, path in index_content]
        counts = Counter(splits)
        logger.info(f"Dataset counts: {counts.most_common()}")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "index_content": [
                        paths for paths, split in zip(index_content, splits) if split == datasets.Split.TRAIN
                    ]
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "index_content": [
                        paths for paths, split in zip(index_content, splits) if split == datasets.Split.VALIDATION
                    ]
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "index_content": [
                        paths for paths, split in zip(index_content, splits) if split == datasets.Split.TEST
                    ]
                },
            ),
        ]

    def _generate_examples(self, index_content: List[Tuple[Path, Path]], **kwargs):
        logger.info(f"Generating {len(index_content)} examples")
        with multiprocessing.Pool(8) as pool:
            for doc_dict in pool.imap(self._try_load_doc, index_content):
                if doc_dict is None:
                    continue
                yield doc_dict["id"], doc_dict

    def _load_index(self, index_path: Path) -> List[Tuple[Path, Path]]:
        raw_content = index_path.read_text()
        lines = raw_content.split("\n")
        rows = [l.split(",") for l in lines]
        return [(Path(r[0]), Path(r[1])) for r in rows if len(r) >= 2]

    def _try_load_doc(self, paths: Tuple[Path, Path]):
        source_path, ibmsg_path = paths
        try:
            doc_dict = self._load_doc(ibmsg_path, source_path)
            doc_dict["id"] = ibmsg_path.stem
            return doc_dict
        except DocLoadingException as e:
            logging.warning(f"Failed to load doc {ibmsg_path}. Full error: {e}")
            return None

    def _load_doc(self, ibmsg_path: Path, image_path: Path):
        content = ibmsg_path.read_bytes()
        fixed_path = self._fix_path(ibmsg_path)

        ibocr, loading_err = ParsedIBOCRBuilder.load_from_str(str(fixed_path), content)
        if loading_err is not None:
            raise DocLoadingException(f"Loading error: {loading_err}")

        try:
            parsed_ibdoc = ibocr.as_parsed_ibocr()
        except Exception as parsing_er:
            raise DocLoadingException(f"Parsing error: {parsing_er}")

        num_records = parsed_ibdoc.get_num_records()
        if num_records != 1:
            raise DocLoadingException(f"Expectected 1 record, got {num_records}")

        record, records_err = parsed_ibdoc.get_ibocr_record(0)
        if records_err is not None:
            raise DocLoadingException(f"Record error: {records_err}")

        validation_err = assert_valid_record(record)
        if validation_err is not None:
            raise DocLoadingException(f"Record validation error: {validation_err}")

        return self._record_to_feature(record, ibmsg_path, image_path)

    def _fix_path(self, ibmsg_path: Path) -> Path:
        # TODO: is it always s1_process_files?
        if "s1_process_files" not in ibmsg_path.parts and "s2_map_records" not in ibmsg_path.parts:
            return ibmsg_path.parent / "s1_process_files" / ibmsg_path.name
        return ibmsg_path

    def _record_to_feature(self, record: IBOCRRecord, ibmsg_path: Path, image_path: Path):
        lines = record.get_lines()
        layouts = [mtd.get_layout() for mtd in record.get_metadata_list()]
        words = []
        word_global_idx = 0
        line_position_to_global_position_map = {}
        for line_idx, line in enumerate(lines):
            for word_idx, word in enumerate(line):
                word["line_index"] = line_idx
                word["word_index"] = word_idx
                word["word_global_idx"] = word_global_idx
                line_position_to_global_position_map[(line_idx, word_idx)] = word_global_idx
                words.append(word)
                word_global_idx += 1

        # get content of the WordPolys
        word_lst: List[str] = [w["word"] for w in words]
        bbox_arr = np.array([[w["start_x"], w["start_y"], w["end_x"], w["end_y"]] for w in words])
        word_pages_arr = np.array([w["page"] for w in words])
        word_line_idx_arr = np.array([w["line_index"] for w in words])
        word_in_line_idx_arr = np.array([w["word_index"] for w in words])

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
        size_per_token = np.take(page_bboxes[:, 2:], word_pages_arr, axis=0)
        fix_bbox_arr = validate_and_fix_bboxes(bbox_arr, size_per_token, word_pages_arr, page_bboxes, ibmsg_path)
        norm_bboxes = fix_bbox_arr * 1000 / size_per_token[:, 0:1]
        norm_page_bboxes = page_bboxes * 1000 / page_bboxes[:, 2:3]

        features = {
            "words": word_lst,
            "bboxes": norm_bboxes,
            "word_original_bboxes": bbox_arr,
            "word_page_nums": word_pages_arr,
            "word_line_idx": word_line_idx_arr,
            "word_in_line_idx": word_in_line_idx_arr,
            "page_bboxes": norm_page_bboxes,
            "page_spans": page_spans,
        }

        if self.config.use_image:
            page_nums = np.unique(word_pages_arr)
            page_nums.sort()
            image = self.image_processor(image_path)
            # assert len(norm_page_bboxes) == len(images), "Number of images should match number of pages in document"
            features["images"] = image[None, :]
            features["images_page_nums"] = page_nums

        return features
