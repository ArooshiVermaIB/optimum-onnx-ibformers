import json
from pathlib import Path
from typing import Any

from ibformers.trainer.ib_package.ModelServiceTemplate.src.py.package_name.model import IbModel
from instabase.protos.model_service import model_service_pb2

dataset_files = Path("/Users/rafalpowalski/python/annotation/UberEatsDataset/out/s1_process_files")

file_list = dataset_files.glob('*ibdoc')


class InstabaseSDKDummy:
    def __init__(self, file_client: Any, username: str):
        # these will be ignored
        self.file_client = file_client
        self.username = username

    def ibopen(self, path: str, mode: str = "r") -> Any:
        return open(path, mode)

    def read_file(self, file_path: str) -> str:
        with open(file_path, 'r') as f:
            return f.read()

    def write_file(self, file_path: str, content: str):
        # mkdir
        Path(file_path).parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, "wb") as f:
            f.write(content)


# Load model
model_path = '/Users/rafalpowalski/python/models/test_model/artifact/src/py/CustomModel/model_data'
sdk = InstabaseSDKDummy(None, "user")
model = IbModel(model_data_path=model_path, ibsdk=sdk)
model.load()

outs = []
for ocr_path in file_list:
    # with open(ocr_path, "rb") as f:
    #     data = f.read()
    req = model_service_pb2.RunModelRequest(input_path=str(ocr_path))
    out = model.run(req)
    outs.append(out)


print(len(outs))
