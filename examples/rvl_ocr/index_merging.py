import logging
from datetime import date
from pathlib import Path
from typing import Optional, List

import fire

from examples.rvl_ocr.config import OcrFlowConfig
from examples.rvl_ocr.image_index import InputDocIndex
from examples.rvl_ocr.ocr_index import OcrIndex
from examples.rvl_ocr.ocr_task import OcrTask

logger = logging.getLogger(__name__)


def get_all_processed_docs(cfg: OcrFlowConfig) -> List[Path]:
    base_documents_path = Path(cfg.output_dir)
    document_paths = []
    document_paths.extend(base_documents_path.glob(f"**/*{cfg.OCR_FLOW_EXTENSION}"))
    return document_paths


def get_image_path(doc_path: Path, input_dir: Path) -> Optional[Path]:
    folder_id, filename = doc_path.stem.split("_", 1)
    dirs = list(folder_id)[:3]
    image_path = input_dir / "images" / f"images{dirs[0]}" / dirs[0] / dirs[1] / dirs[2] / folder_id / filename
    if image_path.exists():
        return image_path
    logger.warning(f"File {image_path} not found")
    return None


def create_ocr_index(config: OcrFlowConfig):
    sub_indices_paths = Path(config.output_dir).glob("*/out_index.txt")
    sub_indices = [OcrIndex.from_file(p) for p in sub_indices_paths]

    merged_index = sum(sub_indices[1:], sub_indices[0])
    num_docs = len(merged_index)

    today = date.today().strftime("%m_%d_%Y")
    index_file_name = f"index_{num_docs}_{today}.txt"
    index_path = Path(config.output_dir) / index_file_name
    if index_path.exists():
        logger.warning(f"Desired index path: {index_path} exists. Remove it before calling the script!")
        return

    merged_index.to_file(index_path)


def add_index_to_old_run(config: OcrFlowConfig):
    input_index = InputDocIndex.read_or_create(config)

    batch_output_dirs = Path(config.output_dir).glob(f"{config.FOLDER_PREFIX}*")
    batch_indices = []
    for batch_output_dir in batch_output_dirs:
        logger.info(f"Creating index for {batch_output_dir}...")
        all_parsed_files = list(batch_output_dir.glob(f"*{config.OCR_FLOW_EXTENSION}"))
        index = OcrIndex(
            original_files=input_index.document_paths,
            ocr_files=all_parsed_files,
            image_files=None,
            unique_name_fn=OcrTask.get_unique_name,
        )
        batch_indices.append(index)
        index.to_file(batch_output_dir / config.BATCH_OUTPUT_INDEX_FILENAME)
    if len(batch_indices) == 0:
        logger.warning(f"No indices found!")
        return

    merged_index = sum(batch_indices[1:], batch_indices[0])
    today = date.today().strftime("%m_%d_%Y")
    index_file_name = f"index_{len(merged_index)}_{today}.txt"
    index_path = Path(config.output_dir) / index_file_name

    if index_path.exists():
        logger.warning(f"Desired index path: {index_path} exists. Remove it before calling the script!")
        return

    merged_index.to_file(index_path)


def main(input_dir: str, output_dir: str):
    config = OcrFlowConfig(input_dir, output_dir)
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    add_index_to_old_run(config)


if __name__ == "__main__":
    fire.Fire(main)
