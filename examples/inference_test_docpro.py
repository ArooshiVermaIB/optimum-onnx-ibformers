import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ibformers.trainer.ib_package.model import IbModel
from instabase.protos.model_service import model_service_pb2
from ocr.client.libs.ibocr import ParsedIBOCRBuilder
from protos.model_service.model_service_pb2 import RawData

dataset_files = Path("/Users/bartosztopolski/fs/driver_licenses/out_annotations/s1_process_files")

file_list = dataset_files.glob("*ibdoc")


class InstabaseSDKDummy:
    def __init__(self, file_client: Any, username: str):
        # these will be ignored
        self.file_client = file_client
        self.username = username

    def ibopen(self, path: str, mode: str = "r") -> Any:
        return open(path, mode)

    def read_file(self, file_path: str) -> str:
        with open(file_path, "r") as f:
            return f.read()

    def write_file(self, file_path: str, content: str):
        # mkdir
        Path(file_path).parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, "wb") as f:
            f.write(content)


# Load model
model_path = "/Users/bartosztopolski/Instabase/models/dls_test"
sdk = InstabaseSDKDummy(None, "user")
model = IbModel(model_data_path=model_path, ibsdk=sdk)
model.load()

outs = []
model_times = []
file_list = list(file_list)
for ocr_path in file_list:
    # with open(ocr_path, "rb") as f:
    #     data = f.read()
    with open(ocr_path, "rb") as f:
        data = f.read()
    ibdoc, err = ParsedIBOCRBuilder.load_from_str(str(ocr_path), data)
    ibdoc.serialize_to_string()
    raw_data = RawData(data=ibdoc.serialize_to_string(), type="ibdoc")
    req = model_service_pb2.RunModelRequest(input_raw_data=raw_data)
    model_start = time.time()
    try:
        out = model.run(req)
        outs.append(out)
    except ValueError as e:
        print(f"Value error: {e}")
    model_end = time.time()
    model_times.append(model_end - model_start)
print(np.mean(model_times))


print(len(outs))
