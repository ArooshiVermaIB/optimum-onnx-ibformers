import asyncio
import logging
import zipfile
from io import BytesIO
from pathlib import Path
import concurrent.futures

from typing import Tuple, Optional

from ci.lib.ibapi import Instabase
from examples.rvl_ocr.config import OcrFlowConfig
from examples.rvl_ocr.task import OcrTask


def get_task_dir_name(cfg: OcrFlowConfig, task: OcrTask) -> str:
    return cfg.FOLDER_PREFIX + str(task.task_id)


def get_task_dir(cfg: OcrFlowConfig, task: OcrTask) -> Path:
    task_dir_name = get_task_dir_name(cfg, task)
    return Path(cfg.output_dir) / task_dir_name


def prepare_task_dir(cfg: OcrFlowConfig, task: OcrTask):
    task_dir = get_task_dir(cfg, task)
    logging.info(f"Preparing task directory {task_dir} for task {task.task_id}. ")
    task_dir.mkdir(exist_ok=True)  # it might be re-started task after failure


def create_zip_file_content(task: OcrTask) -> bytes:
    logging.info(f"Zipping files for task {task.task_id}...")
    bytesio = BytesIO()
    with zipfile.ZipFile(bytesio, "w") as zip_file:
        for file_path in task.file_paths:
            zip_file.write(file_path, file_path.name)
    logging.info(f"Finished zipping files for task {task.task_id}!")
    return bytesio.getvalue()


async def upload_file(cfg: OcrFlowConfig, ibapi: Instabase, task: OcrTask, zip_content: bytes) -> Path:
    ib_path = Path(cfg.ENV_DATA_PATH) / (get_task_dir_name(cfg, task) + '.zip')
    logging.info(f"Uploading files for task {task.task_id} into {ib_path}...")
    await ibapi.write_file(str(ib_path), zip_content)
    logging.info(f"Finished uploading files for task {task.task_id} into {ib_path}...")
    return ib_path


async def unzip_file(cfg: OcrFlowConfig, ibapi: Instabase, zip_ib_path: Path, task: OcrTask) -> Path:
    output_dir = Path(cfg.ENV_DATA_PATH) / zip_ib_path.name.replace('.zip', '')
    logging.info(f"Unzipping files for task {task.task_id} from {zip_ib_path} into {output_dir}...")
    await ibapi.unzip(str(zip_ib_path), str(output_dir))
    return output_dir


async def run_ocr_flow(cfg: OcrFlowConfig, ibapi: Instabase, input_dir: Path) -> Tuple[bool, Optional[str]]:
    logging.info(f"Running flow for {input_dir} directory...")
    status, output_dir = await ibapi.run_flow(
        str(input_dir),
        cfg.ENV_OCR_BINARY_FLOW_PATH,
        True
    )
    logging.info(f"Flow finished with status {status} and output directory {output_dir}")
    return status, output_dir


async def run_task(cfg: OcrFlowConfig, task: OcrTask):
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
    unzipped_doc_dir = await unzip_file(cfg, ibapi, ib_zip_path, task)
    await ibapi.delete_file(str(ib_zip_path))
    flow_status, flow_output_dir = await run_ocr_flow(cfg, ibapi, unzipped_doc_dir)
