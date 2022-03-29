import logging
import multiprocessing
from pathlib import Path

import datasets
from datasets import DatasetInfo, DownloadManager
from typing import Dict, List, Tuple, Any, Sequence, Optional

from ibformers.datasets.ibmsg import ibmsg


logger = logging.getLogger(__name__)

RVL_CLASS_NAMES = [
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific report",
    "scientific publication",
    "specification",
    "file folder",
    "news article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
]


class RVLClassificationConfig(ibmsg.IbmsgConfig):
    def __init__(self, use_image: bool = False, limit_files: Optional[int] = None, num_processes: int = 4, **kwargs):
        super().__init__(use_image=use_image, **kwargs)
        self.limit_files = limit_files
        self.num_processes = num_processes


class RVLClassification(ibmsg.Ibmsg, datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        RVLClassificationConfig(
            name="rvl_classification",
            version=datasets.Version("1.0.0"),
            description="RVL Classification dataset",
        ),
        RVLClassificationConfig(
            name="rvl_classification_5k",
            version=datasets.Version("1.0.0"),
            description="RVL Classification dataset",
            limit_files=5000,
        ),
        RVLClassificationConfig(
            name="rvl_classification_20k",
            version=datasets.Version("1.0.0"),
            description="RVL Classification dataset",
            limit_files=20000,
        ),
    ]

    INDEX_FILENAME = "index.txt"

    TRAIN_LABELS_FILENAME = "labels/train.txt"
    VAL_LABELS_FILENAME = "labels/val.txt"
    TEST_LABELS_FILENAME = "labels/test.txt"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit_files = self.config.limit_files
        self.num_processes = self.config.num_processes

    def _info(self) -> DatasetInfo:
        ds_info = super()._info()
        features = ds_info.features
        features["class_label"] = datasets.features.ClassLabel(names=RVL_CLASS_NAMES)
        return DatasetInfo(
            # This is the description that will appear on the datasets page.
            description="RVL Classification Dataset",
            features=datasets.Features(features),
            supervised_keys=None,
        )

    def _load_label_mapping(self, class_index_path: Path) -> Dict[str, int]:
        index_lines = class_index_path.read_text().split("\n")
        raw_path_class_pairs = (x.split(" ") for x in index_lines if x != "")
        path_class_pairs = map(lambda x: (x[0], int(x[1])), raw_path_class_pairs)
        return dict(path_class_pairs)

    def _load_label_mappings_by_split(self, base_path: Path) -> Dict[str, Dict[str, int]]:
        return {
            "train": self._load_label_mapping(base_path / self.TRAIN_LABELS_FILENAME),
            "val": self._load_label_mapping(base_path / self.VAL_LABELS_FILENAME),
            "test": self._load_label_mapping(base_path / self.TEST_LABELS_FILENAME),
        }

    def _get_ori_file_relative_path(self, ori_file_path: Path) -> str:
        return "/".join(ori_file_path.parts[-6:])

    def _prepare_load_kwargs(
        self, label_mapping: Dict[str, int], index_content: List[Tuple[Path, Path]]
    ) -> Dict[str, Sequence[Any]]:
        maybe_ocr_label_pairs = (
            ((ori_path, ocr_path), label_mapping.get(self._get_ori_file_relative_path(ori_path), None))
            for ori_path, ocr_path in index_content
        )
        ocr_label_pairs = [(paths, label) for paths, label in maybe_ocr_label_pairs if label is not None]
        return {
            "index_content": [paths for paths, label in ocr_label_pairs],
            "labels": [label for paths, label in ocr_label_pairs],
        }

    def _prepare_load_kwargs_by_split(
        self, label_mappings_by_split: Dict[str, Dict[str, int]], index_content: List[Tuple[Path, Path]]
    ) -> Dict[str, Dict[str, Sequence[Any]]]:
        return {
            split_name: self._prepare_load_kwargs(split_label_mapping, index_content)
            for split_name, split_label_mapping in label_mappings_by_split.items()
        }

    def _split_generators(self, dl_manager: DownloadManager):
        if "train" not in self.config.data_files:
            raise ValueError("Please provide a path to the index as train file")
        base_path = Path(self.config.data_files["train"])
        index_path = base_path / self.INDEX_FILENAME
        index_content = self._load_index(index_path)[: self.limit_files]

        label_mappings_by_split = self._load_label_mappings_by_split(base_path)
        load_kwargs_by_split = self._prepare_load_kwargs_by_split(label_mappings_by_split, index_content)

        counts = {split: len(split_data["index_content"]) for split, split_data in load_kwargs_by_split.items()}
        logger.info(f"Dataset counts: {counts}")
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs=load_kwargs_by_split["train"]),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs=load_kwargs_by_split["val"]),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs=load_kwargs_by_split["test"]),
        ]

    def _generate_examples(self, index_content: List[Tuple[Path, Path]], **kwargs):
        labels = kwargs["labels"]
        logger.info(f"Generating {len(index_content)} examples")
        with multiprocessing.Pool(self.num_processes) as pool:
            for doc_dict in pool.imap(self._try_load_doc_with_label, zip(index_content, labels)):
                if doc_dict is None:
                    continue
                yield doc_dict["id"], doc_dict

    def _try_load_doc_with_label(self, paths_and_label: Tuple[Tuple[Path, Path], int]) -> Optional[Dict[str, Any]]:
        paths, label = paths_and_label
        doc = self._try_load_doc(paths)
        if doc is None:
            return None
        doc["class_label"] = label
        return doc
