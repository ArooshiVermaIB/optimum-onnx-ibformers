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
from pathlib import Path
from typing import List, Union

import fire

from ci.lib.config import load_environments_
from ci.lib.ibapi import Instabase
from ibformers.datasets.ibds.ibds import _read_parsedibocr
from instabase.ocr.client.libs.ibocr import ParsedIBOCRBuilder


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
    output_file_path = output_path / Path(file_path).relative_to(root_path)
    output_file_path.parent.mkdir(exist_ok=True, parents=True)
    if binary:
        output_file_path.write_bytes(content)
    else:
        output_file_path.write_text(content)
    return content, output_file_path


async def _download_single_file(
    sdk: Instabase, file_path: str, output_path: Path, relative_root: Path, download_images: bool
):
    file_path = Path(file_path)
    ib_doc_content, output_ibdoc_path = await download_and_save_file(sdk, file_path, relative_root, output_path, True)
    if download_images:
        image_paths = await get_image_paths_from_content(ib_doc_content, output_ibdoc_path)
        image_tasks = [
            download_and_save_file(sdk, image_path, relative_root, output_path, True) for image_path in image_paths
        ]
        await asyncio.gather(*image_tasks, return_exceptions=True)


async def _download_files_from_list(
    sdk: Instabase, file_list: List[Union[Path, str]], output_path: Path, relative_root: Path, download_images: bool
):
    tasks = [
        _download_single_file(sdk, file_path, output_path, relative_root, download_images)
        for file_path in file_list
        if "ibdoc" in str(file_path)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)


async def _ibannotator(sdk: Instabase, input_path: Path, output_path: Path, download_images: bool):
    output_path.mkdir(exist_ok=True, parents=True)
    relative_root = input_path.parent
    annotation_file_content, _ = await download_and_save_file(sdk, input_path, relative_root, output_path, False)
    annotation_dict = json.loads(annotation_file_content)

    file_list = [Path(file["ocrPath"]) for file in annotation_dict["files"]]
    await _download_files_from_list(sdk, file_list, output_path, relative_root, download_images)


async def _ml_studio(sdk: Instabase, input_path: Path, output_path: Path, download_images: bool):
    output_path.mkdir(exist_ok=True, parents=True)
    dataset_path = input_path / "dataset.json"
    edit_info_path = input_path / "edit-info.json"
    relative_root = input_path

    await download_and_save_file(sdk, edit_info_path, relative_root, output_path, False)
    dataset_text, _ = await download_and_save_file(sdk, dataset_path, relative_root, output_path, False)
    dataset_content = json.loads(dataset_text)
    dataset_dir = dataset_content["docs_path"]

    filename_list = await sdk.list_directory(input_path / dataset_dir)
    file_list = [input_path / dataset_dir / f for f in filename_list]
    await _download_files_from_list(sdk, file_list, output_path, relative_root, download_images)


class ProjectDownloader(object):
    def __init__(self, env_name: str):
        envs = load_environments_()

        env_config = envs[env_name]
        self._sdk = Instabase(
            name=env_name,
            host=env_config["host"],
            token=env_config["token"],
            root_path="",
        )

    def ibannotator(self, input_path: str, output_path: str, download_images: bool = False):
        asyncio.run(_ibannotator(self._sdk, Path(input_path), Path(output_path), download_images))

    def ml_studio(self, input_path: Path, output_path: Path, download_images: bool = False):
        asyncio.run(_ml_studio(self._sdk, Path(input_path), Path(output_path), download_images))


if __name__ == "__main__":
    logging.basicConfig(
        level="DEBUG",
        format="%(message)s",
    )

    fire.Fire(ProjectDownloader)
