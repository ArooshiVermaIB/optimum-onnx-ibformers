import dataclasses
import logging
from pathlib import Path
from typing import List

from examples.rvl_ocr.config import OcrFlowConfig


@dataclasses.dataclass
class OcrTask:
    task_id: int
    file_paths: List[Path]


def create_tasks(cfg: OcrFlowConfig, document_index: List[Path]) -> List[OcrTask]:
    logging.info("Creating tasks.")
    tasks = []
    num_docs_per_task = cfg.NUM_DOCS_PER_FOLDER
    for i, start_index in enumerate(range(0, len(document_index), num_docs_per_task)):
        tasks.append(OcrTask(i, document_index[i: (i+num_docs_per_task)]))
    logging.info(f"Created {len(tasks)} tasks with maximum length of {num_docs_per_task}")
    return tasks
