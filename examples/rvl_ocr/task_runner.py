import asyncio
import io
import itertools
import logging
import zipfile
from asyncio import Semaphore
from io import BytesIO
from pathlib import Path
import concurrent.futures

from typing import Tuple, Optional, Dict, List

from ci.lib.ibapi import Instabase
from examples.rvl_ocr.config import OcrFlowConfig
from examples.rvl_ocr.task import OcrTask

logger = logging.getLogger(__name__)


def prepare_task_dir(cfg: OcrFlowConfig, task: OcrTask):
    task_dir = task.get_task_dir(cfg)
    logger.info(f"Task: {task.task_id}: Preparing task directory {task_dir}. ")
    task_dir.mkdir(exist_ok=True)  # it might be re-started task after failure


def create_zip_file_content(task: OcrTask) -> bytes:
    logger.info(f"Task: {task.task_id}: Zipping files...")
    bytesio = BytesIO()
    with zipfile.ZipFile(bytesio, "w") as zip_file:
        for file_path in task.file_paths:
            output_name = task.get_unique_name(file_path)
            zip_file.write(file_path, output_name)
    logger.info(f"Task: {task.task_id}: Finished zipping files!")
    return bytesio.getvalue()


async def upload_file(cfg: OcrFlowConfig, ibapi: Instabase, task: OcrTask, zip_content: bytes) -> Path:
    ib_path = Path(cfg.ENV_DATA_PATH) / (task.get_task_dir_name(cfg) + ".zip")
    logger.info(f"Task: {task.task_id}: Uploading files into {ib_path}...")
    await ibapi.write_file(str(ib_path), zip_content)
    logger.info(f"Task: {task.task_id}: Finished uploading files into {ib_path}...")
    return ib_path


async def unzip_file_on_ib(cfg: OcrFlowConfig, ibapi: Instabase, zip_ib_path: Path, task: OcrTask) -> Path:
    output_dir = Path(cfg.ENV_DATA_PATH) / zip_ib_path.name.replace(".zip", "")
    logger.info(f"Task: {task.task_id}: Unzipping files from {zip_ib_path} into {output_dir}...")
    await ibapi.unzip(str(zip_ib_path), str(output_dir))
    return output_dir


async def run_ocr_flow(
    cfg: OcrFlowConfig, ibapi: Instabase, input_dir: Path, task: OcrTask
) -> Tuple[bool, Optional[str]]:
    logger.info(f"Task: {task.task_id}: Running flow for {input_dir} directory...")
    status, info = await ibapi.run_flow(str(input_dir), cfg.ENV_OCR_BINARY_FLOW_PATH, True)
    if not status:
        logger.warning(f"Task: {task.task_id}: Flow failed for {input_dir} directory...")
        return False, None
    job_id, output_dir = info
    logger.info(f"Task: {task.task_id}: Flow executed with status {status} and output directory {output_dir}")
    await ibapi.wait_for_job_completion(job_id, cfg.FLOW_COMPLETION_WAIT_TIME, True)
    logger.info(f"Task: {task.task_id}: Flow finished for {input_dir}!")
    return status, output_dir


async def get_files_content(cfg: OcrFlowConfig, ibapi: Instabase, task: OcrTask, ib_path: str) -> Dict[Path, bytes]:
    logger.info(f"Task: {task.task_id}: Downloading files from {ib_path} directory...")

    # downloads ocr docs
    ocr_files_dir = Path(ib_path) / cfg.ENV_FLOW_OCR_SUBDIR
    all_files = await ibapi.list_directory(str(ocr_files_dir))
    all_ibdocs = [f for f in all_files if cfg.OCR_FLOW_EXTENSION in str(f)]

    logger.info(f"Task: {task.task_id}: Downloading files.")
    file_tasks = [ibapi.read_binary(str(ocr_files_dir / file)) for file in all_ibdocs]
    contents: List[bytes] = await asyncio.gather(*file_tasks)
    logger.info(f"Task: {task.task_id}: Downloaded files.")

    logger.info(f"Task: {task.task_id}: Finished downloading files from {ib_path} directory")
    return dict(zip([Path(p) for p in all_ibdocs], contents))


def write_files(cfg: OcrFlowConfig, file_contents: Dict[Path, bytes], task: OcrTask):
    logger.info(f"Task: {task.task_id}: writing files...")
    for path, content in file_contents.items():
        out_path = task.get_task_dir(cfg) / path
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_bytes(content)
    logger.info(f"Task: {task.task_id}: finished writing files.")


async def cleanup(cfg: OcrFlowConfig, ibapi: Instabase, task: OcrTask, flow_output_dir: str):
    logger.info(f"Task: {task.task_id}: Cleaning up {flow_output_dir} directory...")
    await ibapi.delete_file(flow_output_dir, recursive=True)
    logger.info(f"Task: {task.task_id}: Cleaned up {flow_output_dir} directory...")


async def run_task(cfg: OcrFlowConfig, task: OcrTask, semaphore: Semaphore, flow_semaphore: Semaphore):
    async with semaphore:
        ibapi = Instabase(
            name=cfg.ENV_NAME,
            host=cfg.ENV_HOST,
            token=cfg.ENV_TOKEN,
            root_path="",
        )
        loop = asyncio.get_running_loop()

        await loop.run_in_executor(None, prepare_task_dir, cfg, task)

        with concurrent.futures.ProcessPoolExecutor(1) as pool:
            zip_file_content = await loop.run_in_executor(pool, create_zip_file_content, task)

        ib_zip_path = await upload_file(cfg, ibapi, task, zip_file_content)
        unzipped_doc_dir = await unzip_file_on_ib(cfg, ibapi, ib_zip_path, task)
        await ibapi.delete_file(str(ib_zip_path))
        async with flow_semaphore:
            flow_status, flow_output_dir = await run_ocr_flow(cfg, ibapi, unzipped_doc_dir, task)
        if flow_status:
            file_content = await get_files_content(cfg, ibapi, task, flow_output_dir)
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                await loop.run_in_executor(pool, write_files, cfg, file_content, task)
            await cleanup(cfg, ibapi, task, flow_output_dir)
            await cleanup(cfg, ibapi, task, str(unzipped_doc_dir))
