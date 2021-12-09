import asyncio
import logging
from asyncio import Semaphore
from pathlib import Path

import fire

from examples.rvl_ocr.config import OcrFlowConfig
from examples.rvl_ocr.image_index import read_or_create_document_index
from examples.rvl_ocr.task import create_tasks
from examples.rvl_ocr.task_runner import run_task


async def run_tasks(cfg: OcrFlowConfig):
    index = read_or_create_document_index(cfg)
    tasks = create_tasks(cfg, index)
    semaphore = Semaphore(cfg.MAX_CONCURRENT_TASKS)
    flow_semaphore = Semaphore(cfg.MAX_CONCURRENT_FLOWS)
    to_run = [run_task(cfg, task, semaphore, flow_semaphore) for task in tasks if not task.is_complete(cfg)]
    await asyncio.gather(*to_run)


def main(input_dir: str, output_dir: str):
    config = OcrFlowConfig(input_dir, output_dir)
    Path(output_dir).mkdir(exist_ok=True)
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(message)s",
    )
    asyncio.get_event_loop().run_until_complete(run_tasks(config))


if __name__ == "__main__":
    fire.Fire(main)
