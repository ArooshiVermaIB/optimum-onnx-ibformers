import dataclasses
import logging
import os
from pathlib import Path

from typing import Optional

import fire


@dataclasses.dataclass
class OcrFlowConfig:
    def __post_init__(self):
        for field in dataclasses.fields(self):
            if field.name not in os.environ:
                continue
            raw_value = os.environ[field.name]
            value = field.type(raw_value)
            logging.info(f"Replacing config field {field.name} with value {value}")
            setattr(self, field.name, value)
    input_dir: str
    output_dir: str

    LOG_LEVEL: str = "INFO"

    ENV_NAME: str = "Azure-OCR"
    ENV_HOST: str = "https://shaunak.azure.sandbox.instabase.com"
    ENV_TOKEN: str = "5xbIePYPc75IjQT9no2zyHpcsXOBf3"
    ENV_DATA_PATH: str = "admin/model-dev/fs/Instabase Drive/OCR_tmp_storage/input"

    ENV_OCR_BINARY_FLOW_PATH: str = "admin/model-dev/fs/Instabase Drive/OCR_tmp_storage/OCR.ibflowbin"

    NUM_DOCS_PER_FOLDER: int = 500
    FOLDER_PREFIX: str = "folder_"

    COMMA_SEPARATED_EXTENSIONS: str = "tif"
    IMAGE_INDEX_FILENAME: str = "index.txt"

    DEBUG_LIMIT_DOC_NUMBER: int = None


def main(input_dir: str, output_dir: str):
    config = OcrFlowConfig(input_dir, output_dir)


if __name__ == "__main__":
    fire.Fire(main)
