import dataclasses
import os


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
    ENV_TOKEN: str = "5xbIePYPc75IjQT9no2zyHpcsXOBf3"
    ENV_DATA_PATH: str = "admin/model-dev/fs/Instabase Drive/OCR_tmp_storage/input"

    ENV_FLOW_OCR_SUBDIR: str = "s1_process_files"
    ENV_FLOW_IMAGE_SUBDIR: str = "images"
    ENV_OCR_BINARY_FLOW_PATH: str = "admin/model-dev/fs/Instabase Drive/OCR_tmp_storage/OCR_and_zip.ibflowbin"
    OCR_FLOW_EXTENSION: str = ".ibmsg"
    ZIPPED_FILE_DIR: str = "s2_reduce_udf"
    ZIPPED_FILE_NAME: str = "zipped.zip"

    NUM_DOCS_PER_FOLDER: int = 500
    FOLDER_PREFIX: str = "folder_"

    COMMA_SEPARATED_EXTENSIONS: str = "tif"
    IMAGE_INDEX_FILENAME: str = "index.txt"

    MAX_CONCURRENT_TASKS: int = 2
    MAX_CONCURRENT_FLOWS: int = 1

    DEBUG_LIMIT_DOC_NUMBER: int = None
