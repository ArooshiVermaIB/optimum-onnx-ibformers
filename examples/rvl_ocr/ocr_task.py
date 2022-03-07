import asyncio
import concurrent.futures
import dataclasses
import logging
import sys
from asyncio import Semaphore
from hashlib import md5
from io import BytesIO
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from sortedcontainers import SortedSet

from ci.lib.ibapi import Instabase
from examples.rvl_ocr.config import OcrFlowConfig
from examples.rvl_ocr.ocr_index import OcrIndex

if sys.version_info >= (3, 8):
    import zipfile
else:
    import zipfile38 as zipfile

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class OcrTask:
    task_no: int
    file_paths: List[Path]
    cfg: OcrFlowConfig

    def __post_init__(self):
        task_hash = md5()
        for path in self.file_paths:
            task_hash.update(str(path).encode())
        self.task_hash = task_hash.hexdigest()[:12]
        self.task_id = f"{self.task_no}-{self.task_hash}"
        self._setup_logging()

    def _setup_logging(self):
        ch = logging.StreamHandler()
        ch.setLevel(self.cfg.LOG_LEVEL)
        formatter = logging.Formatter(f"%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)

        self.logger = logging.getLogger(self.task_id)
        self.logger.addHandler(ch)
        self.logger.setLevel(self.cfg.LOG_LEVEL)
        self.logger.propagate = False

    def get_unique_name(self, file_path: Path):
        if self.cfg.PREPEND_FOLDER_NAME_TO_FILENAME:
            return file_path.parent.name + "_" + file_path.name
        return file_path.name

    def get_output_filename(self, file_path: Path):
        return self.get_unique_name(file_path) + self.cfg.OCR_FLOW_EXTENSION

    def get_task_dir_name(self) -> str:
        return self.cfg.FOLDER_PREFIX + self.task_id

    def get_task_dir(self) -> Path:
        task_dir_name = self.get_task_dir_name()
        return Path(self.cfg.output_dir) / task_dir_name

    def get_post_ocr_index(self) -> Path:
        return self.get_task_dir() / self.cfg.BATCH_OUTPUT_INDEX_FILENAME

    def is_complete(self) -> bool:
        expected_index_path = self.get_post_ocr_index()
        if not expected_index_path.exists():
            self.logger.info("File index not found - marking task as incomplete.")
            return False
        try:
            file_index = OcrIndex.from_file(expected_index_path)
        except:
            self.logger.info("Could not load index file - marking task as incomplete.")
            return False
        already_processed = set(file_index.original_files)
        if set(self.file_paths) != already_processed:
            self.logger.info("Missing files in OCR index - marking task as incomplete.")
            return False
        self.logger.info("Marking task as complete.")
        return True

    async def run_task(self, cfg: OcrFlowConfig, task_semaphore: Semaphore, flow_semaphore: Semaphore):
        async with task_semaphore:
            ibapi = Instabase(
                name=cfg.ENV_NAME,
                host=cfg.ENV_HOST,
                token=cfg.ENV_TOKEN,
                root_path="",
            )
            loop = asyncio.get_running_loop()

            await loop.run_in_executor(None, self.prepare_task_dir)

            with concurrent.futures.ProcessPoolExecutor(1) as pool:
                zip_file_content = await loop.run_in_executor(pool, self.create_zip_file_content)

            ib_zip_path = await self.upload_file(ibapi, zip_file_content)
            unzipped_doc_dir = await self.unzip_file_on_ib(ibapi, ib_zip_path)
            await self.cleanup(ibapi, str(ib_zip_path))
            async with flow_semaphore:
                flow_status, flow_output_dir = await self.run_ocr_flow(ibapi, unzipped_doc_dir)
            if flow_status:
                file_content = await self.get_files_content(ibapi, flow_output_dir)
                images_content = await self.maybe_get_images_content(ibapi, flow_output_dir)
                with concurrent.futures.ThreadPoolExecutor(1) as pool:
                    await loop.run_in_executor(pool, self.write_files, file_content, images_content)
            await self.cleanup(ibapi, str(Path(flow_output_dir).parent.parent))
            await self.cleanup(ibapi, str(unzipped_doc_dir))

    def prepare_task_dir(self):
        task_dir = self.get_task_dir()
        self.logger.info(f"Preparing task directory {task_dir}. ")
        task_dir.mkdir(exist_ok=True)  # it might be re-started task after failure

    def create_zip_file_content(self) -> bytes:
        self.logger.info(f"Zipping files...")
        bytesio = BytesIO()
        with zipfile.ZipFile(bytesio, "w", strict_timestamps=False) as zip_file:
            for file_path in self.file_paths:
                output_name = self.get_unique_name(file_path)
                zip_file.write(file_path, output_name)
        self.logger.info(f"Finished zipping files!")
        return bytesio.getvalue()

    async def upload_file(self, ibapi: Instabase, zip_content: bytes) -> Path:
        ib_path = Path(self.cfg.ENV_DATA_PATH) / (self.get_task_dir_name() + ".zip")
        self.logger.info(f"Uploading files into {ib_path}...")
        await ibapi.write_file(str(ib_path), zip_content)
        self.logger.info(f"Finished uploading files into {ib_path}...")
        return ib_path

    async def unzip_file_on_ib(self, ibapi: Instabase, zip_ib_path: Path) -> Path:
        output_dir = Path(self.cfg.ENV_DATA_PATH) / zip_ib_path.name.replace(".zip", "")
        self.logger.info(f"Unzipping files from {zip_ib_path} into {output_dir}...")
        await ibapi.unzip(str(zip_ib_path), str(output_dir))
        return output_dir

    async def run_ocr_flow(self, ibapi: Instabase, input_dir: Path) -> Tuple[bool, Optional[str]]:
        self.logger.info(f"Running flow for {input_dir} directory...")
        status, info = await ibapi.run_flow(str(input_dir), self.cfg.ENV_OCR_BINARY_FLOW_PATH, True)
        if not status:
            logger.warning(f"Task: {self.task_id}: Flow failed for {input_dir} directory...")
            return False, None
        job_id, output_dir = info
        self.logger.info(f"Flow executed with status {status} and output directory {output_dir}")
        await ibapi.wait_for_job_completion(job_id, self.cfg.FLOW_COMPLETION_WAIT_TIME, True)
        self.logger.info(f"Flow finished for {input_dir}!")
        return status, output_dir

    async def get_files_content(self, ibapi: Instabase, ib_path: str) -> Dict[str, bytes]:
        self.logger.info(f"Downloading files from {ib_path} directory...")

        # downloads ocr docs
        ocr_files_dir = Path(ib_path) / self.cfg.ENV_FLOW_OCR_SUBDIR
        all_files = await ibapi.list_directory(str(ocr_files_dir))
        all_ibdocs = [f for f in all_files if self.cfg.OCR_FLOW_EXTENSION in str(f)]

        self.logger.info(f"Downloading files.")
        file_tasks = [ibapi.read_binary(str(ocr_files_dir / file)) for file in all_ibdocs]
        contents: List[bytes] = await asyncio.gather(*file_tasks)  # type: ignore
        self.logger.info(f"Downloaded files.")

        self.logger.info(f"Finished downloading files from {ib_path} directory")
        return dict(zip([str(Path(self.cfg.ENV_FLOW_OCR_SUBDIR) / p) for p in all_ibdocs], contents))

    async def maybe_get_images_content(self, ibapi: Instabase, ib_path: str) -> Dict[str, bytes]:
        if not self.cfg.DOWNLOAD_IMAGES:
            self.logger.debug(f"Skipping downloading images from {ib_path} directory...")
            return {}
        self.logger.info(f"Downloading images from {ib_path} directory...")

        images_dir = Path(ib_path) / self.cfg.IMAGES_SUBDIR
        all_images = await ibapi.list_directory(str(images_dir))

        self.logger.info(f"Downloading images...")
        file_tasks = [ibapi.read_binary(str(images_dir / file)) for file in all_images]
        contents: List[bytes] = await asyncio.gather(*file_tasks)  # type: ignore
        self.logger.info(f"Downloaded images...")

        self.logger.info(f"Finished downloading images from {ib_path} directory")
        return dict(zip([str(Path(self.cfg.IMAGES_SUBDIR) / p) for p in all_images], contents))

    def write_files(self, file_contents: Dict[str, bytes], images_contents: Dict[str, bytes]):
        self.write_downloaded_files(dict(**file_contents, **images_contents))
        self.write_config()
        ocr_files = [self.get_task_dir() / ocr_path for ocr_path in file_contents.keys()]
        index = OcrIndex(self.file_paths, ocr_files, None, False, self.get_unique_name)
        index.to_file(self.get_task_dir() / self.cfg.BATCH_OUTPUT_INDEX_FILENAME)

    def write_downloaded_files(self, file_contents: Dict[str, bytes]):
        self.logger.info(f"writing files...")
        for path, content in file_contents.items():
            out_path = self.get_task_dir() / path
            out_path.parent.mkdir(exist_ok=True, parents=True)
            out_path.write_bytes(content)
        self.logger.info(f"finished writing files.")

    def write_config(self):
        config_path = self.get_task_dir() / "ocr_config.json"
        self.cfg.save_json(config_path)

    async def cleanup(self, ibapi: Instabase, flow_output_dir: str):
        self.logger.info(f"Cleaning up {flow_output_dir} directory...")
        await ibapi.delete_file(flow_output_dir, recursive=True)
        self.logger.info(f"Cleaned up {flow_output_dir} directory...")


def get_existing_task_directories(cfg: OcrFlowConfig):
    all_subdirs = list(Path(cfg.output_dir).glob("*"))
    return [subdir for subdir in all_subdirs if subdir.is_dir() and subdir.name.startswith(cfg.FOLDER_PREFIX)]


def get_existing_index_files(cfg: OcrFlowConfig, ocr_task_subdirs: List[Path]) -> List[Path]:
    return [
        p / cfg.BATCH_OUTPUT_INDEX_FILENAME for p in ocr_task_subdirs if (p / cfg.BATCH_OUTPUT_INDEX_FILENAME).exists()
    ]


def get_first_task_id(cfg: OcrFlowConfig, ocr_task_subdirs: List[Path]) -> int:
    all_ids = [int(subdir.name.replace(cfg.FOLDER_PREFIX, "").split("-")[0]) for subdir in ocr_task_subdirs]
    if len(all_ids) == 0:
        return 0
    return max(all_ids) + 1


def prepare_for_task_creation(cfg: OcrFlowConfig, document_index: List[Path]) -> Tuple[List[Path], int]:
    logger.info("Checking for already processed files...")
    existing_subdirs = get_existing_task_directories(cfg)
    existing_index_files = get_existing_index_files(cfg, existing_subdirs)
    loaded_indices = [OcrIndex.from_file(p) for p in existing_index_files]
    if len(loaded_indices) > 0:
        merged_index = sum(loaded_indices[1:], loaded_indices[0])
        existing_files = set(merged_index.original_files)
        unprocessed_files = [i for i in document_index if i not in existing_files]
    else:
        unprocessed_files = document_index
    first_task_id = get_first_task_id(cfg, existing_subdirs)
    logger.info(
        f"Selected {len(unprocessed_files)} out of {len(document_index)} original files. The "
        f"first batch index will be {first_task_id}"
    )
    return unprocessed_files, first_task_id


def create_tasks(cfg: OcrFlowConfig, document_index: List[Path]) -> List[OcrTask]:
    logger.info("Creating tasks.")
    tasks = []

    files_to_process, first_index = prepare_for_task_creation(cfg, document_index)

    num_docs_per_task = cfg.NUM_DOCS_PER_FOLDER
    for i, start_index in enumerate(range(0, len(files_to_process), num_docs_per_task), first_index):
        tasks.append(OcrTask(i, files_to_process[start_index : (start_index + num_docs_per_task)], cfg))
    logger.info(f"Created {len(tasks)} tasks with maximum length of {num_docs_per_task}")
    return tasks
