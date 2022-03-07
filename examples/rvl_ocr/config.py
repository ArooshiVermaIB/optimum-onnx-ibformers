import dataclasses
import json
import os
from pathlib import Path


@dataclasses.dataclass
class OcrFlowConfig:
    def __post_init__(self):
        for field in dataclasses.fields(self):
            if field.name not in os.environ:
                continue
            raw_value = os.environ[field.name]
            value = field.type(raw_value)
            setattr(self, field.name, value)

    input_dir: str
    output_dir: str

    LOG_LEVEL: str = "INFO"

    ENV_NAME: str = "Azure-OCR"
    ENV_HOST: str = "https://shaunak.azure.sandbox.instabase.com"
    ENV_TOKEN: str = "ibZB13GIcaWfMT3LuhHDYcxsCFMPPQ"
    ENV_DATA_PATH: str = "topolskib/my-repo/fs/Instabase Drive/OCR_tmp_storage/input"

    ENV_FLOW_OCR_SUBDIR: str = "s2_map_records"
    ENV_OCR_BINARY_FLOW_PATH: str = "topolskib/my-repo/fs/Instabase Drive/OCR_tmp_storage/OCR_map_records.ibflowbin"
    OCR_FLOW_EXTENSION: str = ".ibmsg"

    DOWNLOAD_IMAGES: int = 1
    IMAGES_SUBDIR = "s1_process_files/images"

    FLOW_COMPLETION_WAIT_TIME: int = 30
    NUM_DOCS_PER_FOLDER: int = 200
    FOLDER_PREFIX: str = "folder_"

    PREPEND_FOLDER_NAME_TO_FILENAME: int = 1
    COMMA_SEPARATED_EXTENSIONS: str = "tif"
    IMAGE_INDEX_FILENAME: str = "index.txt"
    BATCH_OUTPUT_INDEX_FILENAME: str = "out_index.txt"
    MAX_NUM_PAGES: int = 10
    MAX_FILE_SIZE_IN_KB: int = 1024
    SORT_INDEX_BY_NUM_PAGES: int = 1

    MAX_CONCURRENT_TASKS: int = 7
    MAX_CONCURRENT_FLOWS: int = 6
    MIN_DELAY_BETWEEN_JOBS: int = 15

    DEBUG_LIMIT_DOC_NUMBER: int = None

    def save_json(self, output_path: Path):
        config_dict = dataclasses.asdict(self)
        output_path.write_text(json.dumps(config_dict, indent=2))
