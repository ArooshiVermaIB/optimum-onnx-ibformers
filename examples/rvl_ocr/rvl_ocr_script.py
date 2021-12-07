import asyncio
import logging
from pathlib import Path

import fire

from examples.rvl_ocr.config import OcrFlowConfig
from examples.rvl_ocr.image_index import read_or_create_document_index
from examples.rvl_ocr.task import create_tasks
from examples.rvl_ocr.task_runner import run_task


async def run_tests(cfg: OcrFlowConfig):
    index = read_or_create_document_index(cfg)
    tasks = create_tasks(cfg, index)
    to_run = [run_task(cfg, task) for task in tasks[:2]]
    await asyncio.gather(*to_run)


def main(input_dir: str, output_dir: str):
    config = OcrFlowConfig(input_dir, output_dir)
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(message)s",
    )
    asyncio.get_event_loop().run_until_complete(
        run_tests(config)
    )


if __name__ == "__main__":
    fire.Fire(main)
