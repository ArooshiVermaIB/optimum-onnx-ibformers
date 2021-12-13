import csv
import logging
from datetime import date
from pathlib import Path

from typing import Optional, List

import fire

from examples.rvl_ocr.config import OcrFlowConfig


def get_all_processed_docs(cfg: OcrFlowConfig) -> List[Path]:
    base_documents_path = Path(cfg.output_dir)
    document_paths = []
    document_paths.extend(base_documents_path.glob(f"**/*{cfg.OCR_FLOW_EXTENSION}"))
    return document_paths


def get_image_path(doc_path: Path, input_dir: Path) -> Optional[Path]:
    folder_id, filename = doc_path.stem.split("_", 1)
    dirs = list(folder_id)[:3]
    image_path = input_dir / f"images{dirs[0]}" / dirs[0] / dirs[1] / dirs[2] / folder_id / filename
    if image_path.exists():
        return image_path
    logging.warning(f"File {image_path} not found")
    return None


def create_post_ocr_index(config: OcrFlowConfig):
    doc_paths = get_all_processed_docs(config)
    input_dir = Path(config.input_dir)
    pairs = [(p, get_image_path(p, input_dir)) for p in doc_paths]
    valid_pairs = [pair for pair in pairs if pairs[1] is not None]

    today = date.today().strftime("%m_%d_%Y")
    index_file_name = f"index_{len(valid_pairs)}_{today}.txt"
    index_path = Path(config.output_dir) / index_file_name
    if index_path.exists():
        logging.warning(f"Desired index path: {index_path} exists. Remove it before calling the script!")
        return

    with index_path.open(mode="w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        for pair in valid_pairs:
            writer.writerow(pair)


def main(input_dir: str, output_dir: str):
    config = OcrFlowConfig(input_dir, output_dir)
    create_post_ocr_index(config)


if __name__ == "__main__":
    fire.Fire(main)
