import asyncio
import logging
from asyncio import Semaphore
from pathlib import Path

import fire

from examples.rvl_ocr.async_utils import SemaphoreWithDelay
from examples.rvl_ocr.config import OcrFlowConfig
from examples.rvl_ocr.image_index import InputDocIndex
from examples.rvl_ocr.index_merging import create_ocr_index
from examples.rvl_ocr.ocr_task import create_tasks


async def run_tasks(cfg: OcrFlowConfig, return_exceptions: bool):
    index = InputDocIndex.read_or_create(cfg)
    valid_doc_paths = index.sort_and_filter_paths(cfg)[: cfg.DEBUG_LIMIT_DOC_NUMBER]
    tasks = create_tasks(cfg, valid_doc_paths)
    semaphore = Semaphore(cfg.MAX_CONCURRENT_TASKS)
    flow_semaphore = SemaphoreWithDelay(cfg.MAX_CONCURRENT_FLOWS, cfg.MIN_DELAY_BETWEEN_JOBS)
    to_run = [task.run_task(cfg, semaphore, flow_semaphore) for task in tasks]
    await asyncio.gather(*to_run, return_exceptions=return_exceptions)

    create_ocr_index(cfg)
    config_path = Path(cfg.output_dir) / "ocr_config.json"
    cfg.save_json(config_path)


def main(input_dir: str, output_dir: str, return_exceptions: bool = True):
    config = OcrFlowConfig(input_dir, output_dir)
    Path(output_dir).mkdir(exist_ok=True)
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.get_event_loop().run_until_complete(run_tasks(config, return_exceptions=return_exceptions))


if __name__ == "__main__":
    fire.Fire(main)
