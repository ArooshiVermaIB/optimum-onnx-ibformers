import csv
import logging
from pathlib import Path
from typing import List, Callable

logger = logging.getLogger(__name__)


class OcrIndex:
    def __init__(
        self,
        original_files: List[Path],
        ocr_files: List[Path],
        image_files: List[Path] = None,
        are_matched: bool = False,
        unique_name_fn: Callable[[Path], str] = lambda x: x.name,
    ):
        skip_images = image_files is None
        if are_matched:
            self.original_files = original_files
            self.ocr_files = ocr_files
            self.image_files = image_files
        else:
            self.original_files = []
            self.ocr_files = []
            ocr_names = {ocr_file.stem: ocr_file for ocr_file in ocr_files}

            if not skip_images:
                self.image_files = []
                image_names = {image_file.stem: image_file for image_file in image_files}
                raw_image_paths = set(str(image_file) for image_file in image_files)
            for ori_file in original_files:
                unique_name = unique_name_fn(ori_file)
                matched_ocr_file = ocr_names.get(unique_name, None)

                if not skip_images:
                    matched_image_file = image_names.get(unique_name, None)
                    if matched_image_file is None:
                        matched_image_file = ori_file if ori_file in raw_image_paths else None
                else:
                    matched_image_file = ""

                if matched_ocr_file is None or matched_image_file is None:
                    continue
                self.original_files.append(ori_file)
                self.ocr_files.append(matched_ocr_file)
                if not skip_images:
                    self.image_files.append(matched_image_file)

        if skip_images:
            self.image_files = self.original_files
        assert len(self.original_files) == len(self.ocr_files) == len(self.image_files)

    def __len__(self):
        return len(self.original_files)

    def __getitem__(self, item):
        return [self.original_files[item], self.ocr_files[item], self.image_files[item]]

    def __add__(self, other: "OcrIndex") -> "OcrIndex":
        return OcrIndex(
            self.original_files + other.original_files,
            self.ocr_files + other.ocr_files,
            self.image_files + other.image_files,
            are_matched=True,
        )

    def to_file(self, output_path: Path):
        with output_path.open(mode="w", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            for ori_path, ocr_path, image_path in self:
                writer.writerow((str(ori_path), str(ocr_path), str(image_path)))

    @classmethod
    def from_file(cls, input_path: Path):
        with input_path.open(mode="r", newline="") as f:
            reader = csv.reader(f, delimiter=",")
            original_files, ocr_files, image_files = zip(*reader)
        return cls(
            [Path(ori_file) for ori_file in original_files],
            [Path(ocr_file) for ocr_file in ocr_files],
            [Path(image_file) for image_file in image_files],
            True,
        )
