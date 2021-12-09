import dataclasses
import logging
from pathlib import Path
from typing import List

from examples.rvl_ocr.config import OcrFlowConfig


@dataclasses.dataclass
class OcrTask:
    task_id: int
    file_paths: List[Path]

    @staticmethod
    def get_unique_name(file_path: Path):
        return file_path.parent.name + "_" + file_path.name

    def get_task_dir_name(self, cfg: OcrFlowConfig) -> str:
        return cfg.FOLDER_PREFIX + str(self.task_id)

    def get_task_dir(self, cfg: OcrFlowConfig) -> Path:
        task_dir_name = self.get_task_dir_name(cfg)
        return Path(cfg.output_dir) / task_dir_name

    def is_complete(self, cfg: OcrFlowConfig) -> bool:
        task_dir = self.get_task_dir(cfg)
        for file_path in self.file_paths:
            name = self.get_unique_name(file_path) + cfg.OCR_FLOW_EXTENSION
            if not (task_dir / name).exists():
                return False
        logging.info(f"Marking task {self.task_id} as complete.")
        return True


def create_tasks(cfg: OcrFlowConfig, document_index: List[Path]) -> List[OcrTask]:
    logging.info("Creating tasks.")
    tasks = []
    num_docs_per_task = cfg.NUM_DOCS_PER_FOLDER
    for i, start_index in enumerate(range(0, len(document_index), num_docs_per_task)):
        tasks.append(OcrTask(i, document_index[i : (i + num_docs_per_task)]))
    logging.info(f"Created {len(tasks)} tasks with maximum length of {num_docs_per_task}")
    return tasks
