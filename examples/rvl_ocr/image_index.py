import logging
from pathlib import Path

from typing import List

from examples.rvl_ocr.config import OcrFlowConfig


def create_document_index(cfg: OcrFlowConfig) -> List[Path]:
    base_documents_path = Path(cfg.input_dir)
    index_path = base_documents_path / cfg.IMAGE_INDEX_FILENAME
    document_paths = []
    for extension in cfg.COMMA_SEPARATED_EXTENSIONS.split(","):
        document_paths.extend(base_documents_path.glob(f"**/*.{extension}"))
    index_path.write_text("\n".join(map(str, document_paths)))
    logging.info(f"Index created at {index_path}. {len(document_paths)} documents found in the index..")
    return document_paths


def read_or_create_document_index(cfg: OcrFlowConfig) -> List[Path]:
    base_documents_path = Path(cfg.input_dir)
    index_path = base_documents_path / cfg.IMAGE_INDEX_FILENAME
    if index_path.exists():
        logging.info(f"Index found at {index_path}. Reading its contents.")
        path_list = (base_documents_path / cfg.IMAGE_INDEX_FILENAME).read_text().split("\n")
        image_index = [Path(path) for path in path_list]
        logging.info(f"{len(image_index)} documents found in the index.")
    else:
        logging.info(f"Index not found at {index_path}. Creating new index.")
        image_index = create_document_index(cfg)
    if cfg.DEBUG_LIMIT_DOC_NUMBER is not None:
        return image_index[: cfg.DEBUG_LIMIT_DOC_NUMBER]
    return image_index
