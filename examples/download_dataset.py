"""
Script for downloading dataset from instabase instances.

It downloads all necessary files, including images from ibdoc files. DOES NOT change the path to the image
after downloading, but preserves the directory structure and the relative paths.

Example usage:
python download_dataset.py ml_studio \
  --env_name dogfood --input_path path/to/my/mlstudio/ds --output_path /path/to/save/dataset/to
"""
import asyncio
import json
import logging
from asyncio import Semaphore
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Tuple

import fire

from ci.lib.config import load_environments
from ci.lib.ibapi import Instabase
from instabase.ocr.client.libs.ibocr import (
    ParsedIBOCRBuilder,
    IBOCRRecord,
    IBOCRRecordLayout,
    ParsedIBOCR,
)
from instabase.ocr.client.libs.ocr_types import WordPolyDict


@dataclass
class DownloaderConfig:
    input_path: Path
    output_path: Path
    download_images: bool
    max_concurrent_downloads: int

    def __post_init__(self):
        self.download_semaphore = Semaphore(self.max_concurrent_downloads)


def _read_parsedibocr(builder: ParsedIBOCR) -> Tuple[List[WordPolyDict], List[IBOCRRecordLayout]]:
    """Open an ibdoc or ibocr using the ibfile and return the words and layout information for each page"""
    words = []
    layouts = []
    record: IBOCRRecord
    # Assuming each record is a page in order and each record is single-page
    # Assuming nothing weird is going on with page numbers
    for record in builder.get_ibocr_records():
        words += [i for j in record.get_lines() for i in j]
        l = record.get_metadata_list()
        layouts.extend([i.get_layout() for i in l])

    assert all(word["page"] in range(len(layouts)) for word in words), "Something with the page numbers went wrong"

    return words, layouts


async def get_image_paths_from_content(content: bytes, save_path: Path) -> List[Path]:
    builder, err = ParsedIBOCRBuilder.load_from_str(str(save_path), content)
    ibocr = builder.as_parsed_ibocr()
    words, layouts = _read_parsedibocr(ibocr)
    return [Path(layout.get_processed_image_path()) for layout in layouts]


async def download_and_save_file(sdk: Instabase, file_path: Path, root_path: Path, output_path: Path, binary: bool):
    if binary:
        content = await sdk.read_binary(file_path)
    else:
        content = await sdk.read_file(file_path)
    if "out_annotations" in Path(file_path).parts:
        idx = Path(file_path).parts.index("out_annotations")
        relative_path = Path(*Path(file_path).parts[idx:])
    else:
        relative_path = Path(file_path).relative_to(root_path)
    output_file_path = output_path / relative_path
    output_file_path.parent.mkdir(exist_ok=True, parents=True)
    if binary:
        output_file_path.write_bytes(content)
    else:
        output_file_path.write_text(content)
    return content, output_file_path


async def _download_single_ibdoc(sdk: Instabase, cfg: DownloaderConfig, file_path: str, relative_root: Path):
    async with cfg.download_semaphore:
        file_path = Path(file_path)
        ib_doc_content, output_ibdoc_path = await download_and_save_file(
            sdk, file_path, relative_root, cfg.output_path, True
        )
    if cfg.download_images:
        async with cfg.download_semaphore:
            image_paths = await get_image_paths_from_content(ib_doc_content, output_ibdoc_path)
            for image_path in image_paths:
                await download_and_save_file(sdk, image_path, relative_root, cfg.output_path, True)


async def _download_files_from_list(
    sdk: Instabase, cfg: DownloaderConfig, file_list: List[Union[Path, str]], relative_root: Path
):
    tasks = [download_and_save_file(sdk, file_path, relative_root, cfg.output_path, True) for file_path in file_list]
    await asyncio.gather(*tasks, return_exceptions=False)


async def _download_ibdocs_from_list(
    sdk: Instabase, cfg: DownloaderConfig, file_list: List[Union[Path, str]], relative_root: Path
):
    tasks = [
        _download_single_ibdoc(sdk, cfg, file_path, relative_root)
        for file_path in file_list
        if "ibdoc" in str(file_path)
    ]
    await asyncio.gather(*tasks, return_exceptions=False)


async def _ibannotator(sdk: Instabase, cfg: DownloaderConfig):
    cfg.output_path.mkdir(exist_ok=True, parents=True)
    relative_root = cfg.input_path.parent
    annotation_file_content, _ = await download_and_save_file(
        sdk, cfg.input_path, relative_root, cfg.output_path, False
    )
    annotation_dict = json.loads(annotation_file_content)

    file_list = [Path(file["ocrPath"]) for file in annotation_dict["files"]]
    await _download_ibdocs_from_list(sdk, cfg, file_list, relative_root)


async def _ml_studio(sdk: Instabase, cfg: DownloaderConfig):
    cfg.output_path.mkdir(exist_ok=True, parents=True)
    dataset_path = cfg.input_path / "dataset.json"
    edit_info_path = cfg.input_path / "edit-info.json"
    relative_root = cfg.input_path

    await download_and_save_file(sdk, edit_info_path, relative_root, cfg.output_path, False)
    dataset_text, _ = await download_and_save_file(sdk, dataset_path, relative_root, cfg.output_path, False)
    dataset_content = json.loads(dataset_text)
    dataset_dir = dataset_content["docs_path"]
    annotations_dir = dataset_content["annotations_folder_path"]
    if annotations_dir is None:
        annotations_dir = "out_annotations/annotations"

    annotations_list = await sdk.list_directory(cfg.input_path / annotations_dir)
    anno_file_list = [cfg.input_path / annotations_dir / f for f in annotations_list]
    await _download_files_from_list(sdk, cfg, anno_file_list, relative_root)

    filename_list = await sdk.list_directory(cfg.input_path / dataset_dir)
    file_list = [cfg.input_path / dataset_dir / f for f in filename_list]
    await _download_ibdocs_from_list(sdk, cfg, file_list, relative_root)


async def _directory(sdk: Instabase, cfg: DownloaderConfig):
    cfg.output_path.mkdir(exist_ok=True, parents=True)
    relative_root = cfg.input_path
    filename_list = await sdk.list_directory(str(cfg.input_path))
    file_list = [cfg.input_path / f for f in filename_list]
    await _download_ibdocs_from_list(sdk, cfg, file_list, relative_root)


class ProjectDownloader(object):
    def __init__(
        self,
        env_name: str,
        input_path: str,
        output_path: str,
        download_images: bool = False,
        max_concurrent_downloads: int = 10,
        token: str = None,
    ):
        envs = load_environments()
        env_config = envs[env_name]
        token = env_config["token"] if token is None else token
        self._sdk = Instabase(
            name=env_name,
            host=env_config["host"],
            token=token,
            root_path="",
        )

        self.config = DownloaderConfig(Path(input_path), Path(output_path), download_images, max_concurrent_downloads)

    def ibannotator(self):
        asyncio.get_event_loop().run_until_complete(_ibannotator(self._sdk, self.config))

    def ml_studio(self):
        asyncio.get_event_loop().run_until_complete(_ml_studio(self._sdk, self.config))

    def directory(self):
        asyncio.get_event_loop().run_until_complete(_directory(self._sdk, self.config))


if __name__ == "__main__":
    logging.basicConfig(
        level="DEBUG",
        format="%(message)s",
    )

    fire.Fire(ProjectDownloader)
