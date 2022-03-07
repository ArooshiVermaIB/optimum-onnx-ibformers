import csv
import logging
from pathlib import Path

from typing import List, Tuple

import PIL
from PIL import Image

from examples.rvl_ocr.config import OcrFlowConfig

logger = logging.getLogger(__name__)


INVALID_FILE_NUM_PAGES = 0

IMAGE_EXTENSIONS = [
    ".blp",
    ".bmp",
    ".dib",
    ".bufr",
    ".cur",
    ".pcx",
    ".dcx",
    ".dds",
    ".ps",
    ".eps",
    ".fit",
    ".fits",
    ".fli",
    ".flc",
    ".fpx",
    ".ftc",
    ".ftu",
    ".gbr",
    ".gif",
    ".grib",
    ".h5",
    ".hdf",
    ".png",
    ".apng",
    ".jp2",
    ".j2k",
    ".jpc",
    ".jpf",
    ".jpx",
    ".j2c",
    ".icns",
    ".ico",
    ".im",
    ".iim",
    ".tif",
    ".tiff",
    ".jfif",
    ".jpe",
    ".jpg",
    ".jpeg",
    ".mic",
    ".mpg",
    ".mpeg",
    ".mpo",
    ".msp",
    ".palm",
    ".pcd",
    ".pdf",
    ".pxr",
    ".pbm",
    ".pgm",
    ".ppm",
    ".pnm",
    ".psd",
    ".bw",
    ".rgb",
    ".rgba",
    ".sgi",
    ".ras",
    ".tga",
    ".icb",
    ".vda",
    ".vst",
    ".webp",
    ".wmf",
    ".emf",
    ".xbm",
    ".xpm",
]


def get_image_num_pages(image_path: Path) -> int:
    try:
        img = Image.open(image_path)
        num_pages = img.n_frames
        img.close()
    except PIL.UnidentifiedImageError:
        return INVALID_FILE_NUM_PAGES
    except:
        logger.error(
            f"Failed to read number of pages from {image_path}. Marking it as invalid by setting "
            f"number of pages to {INVALID_FILE_NUM_PAGES}"
        )
        return INVALID_FILE_NUM_PAGES
    return num_pages


def get_num_pages(image_path: Path) -> int:
    if image_path.suffix in IMAGE_EXTENSIONS:
        return get_image_num_pages(image_path)
    logger.warning(
        f"Getting number of pages from {image_path.suffix} extension is not implemented. " f"Assuming single page."
    )
    return 1


class InputDocIndex:
    def __init__(
        self,
        document_paths: List[Path],
        file_sizes: List[float] = None,
        page_counts: List[int] = None,
        needs_saving: bool = False,
    ):
        if file_sizes is not None:
            assert len(document_paths) == len(file_sizes)
        if page_counts is not None:
            assert len(document_paths) == len(page_counts)

        self.document_paths = document_paths
        self.file_sizes = file_sizes
        self.page_counts = page_counts
        self.needs_saving = needs_saving

    def __getitem__(self, item) -> Tuple[Path, float, int]:
        return self.document_paths[item], self.file_sizes[item], self.page_counts[item]

    @staticmethod
    def get_document_paths(cfg: OcrFlowConfig) -> List[Path]:
        base_documents_path = Path(cfg.input_dir)
        logger.info(f"Searching for files [{cfg.COMMA_SEPARATED_EXTENSIONS}] in {base_documents_path}")
        document_paths = []
        for extension in cfg.COMMA_SEPARATED_EXTENSIONS.split(","):
            document_paths.extend(base_documents_path.glob(f"**/*.{extension}"))
        return document_paths

    @staticmethod
    def get_file_sizes(document_paths: List[Path]) -> List[float]:
        return [doc_path.stat().st_size / 1024 for doc_path in document_paths]

    @staticmethod
    def get_page_counts(document_paths: List[Path]) -> List[int]:
        return [get_num_pages(doc_path) for doc_path in document_paths]

    @classmethod
    def create_from_config(cls, cfg: OcrFlowConfig):
        document_paths = cls.get_document_paths(cfg)
        file_sizes = cls.get_file_sizes(document_paths)
        page_counts = cls.get_page_counts(document_paths)
        logger.info(f"Index created from {cfg.input_dir}. {len(document_paths)} documents found in the index..")
        return cls(document_paths, file_sizes, page_counts, True)

    @classmethod
    def from_file(cls, input_path: Path):
        logger.info(f"Loading index from file {input_path}")
        with input_path.open(mode="r", newline="") as f:
            reader = csv.reader(f, delimiter=",")
            document_paths, *other_cols = zip(*reader)
            document_paths = [Path(d) for d in document_paths]
        if len(other_cols) == 0:
            needs_saving = True
            logger.info(f"File sizes and page counts not found in {input_path}. Calculating...")
            file_sizes = cls.get_file_sizes(document_paths)
            page_counts = cls.get_page_counts(document_paths)
        elif len(other_cols) == 2:
            needs_saving = False
            file_sizes, page_counts = other_cols
            file_sizes = [float(s) for s in file_sizes]
            page_counts = [int(p) for p in page_counts]
        else:
            raise ValueError(f"Invalid format for index {input_path}")
        logger.info(f"Loaded index from file {input_path}. {len(document_paths)} documents found in the index..")
        return cls(document_paths, file_sizes, page_counts, needs_saving)

    def to_file(self, output_path: Path):
        logger.info(f"Saving index to file {output_path}")
        with output_path.open(mode="w", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            for doc_path, file_size, num_pages in self:
                writer.writerow((str(doc_path), str(file_size), str(num_pages)))

    @classmethod
    def read_or_create(cls, cfg: OcrFlowConfig):
        base_documents_path = Path(cfg.input_dir)
        index_path = base_documents_path / cfg.IMAGE_INDEX_FILENAME
        if index_path.exists():
            index = cls.from_file(index_path)
        else:
            index = cls.create_from_config(cfg)
        if index.needs_saving:
            index.to_file(index_path)
        return index

    def sort_and_filter_paths(self, cfg: OcrFlowConfig) -> List[Path]:
        logger.info(f"Filtering document list...")
        index = self
        if cfg.MAX_NUM_PAGES is not None:
            logger.info(f"Removing documents with more pages than {cfg.MAX_NUM_PAGES}")
            index = filter(lambda x: 0 < x[2] <= cfg.MAX_NUM_PAGES, index)
        if cfg.MAX_FILE_SIZE_IN_KB is not None:
            logger.info(f"Removing documents with larger size than {cfg.MAX_FILE_SIZE_IN_KB} kb")
            index = filter(lambda x: 0 < x[1] <= cfg.MAX_FILE_SIZE_IN_KB, index)
        if cfg.SORT_INDEX_BY_NUM_PAGES:
            logger.info(f"Sorting documents by number of pages and size...")
            index = sorted(index, key=lambda x: (x[2], x[1]))
        logger.info(f"{len(index)} documents left after the filtering.")
        return [t[0] for t in index]
