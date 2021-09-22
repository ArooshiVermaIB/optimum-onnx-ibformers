import json
from pathlib import Path
from typing import Any

from instabase.ocr.client.libs.ibocr import ParsedIBOCRBuilder, IBOCRRecord, IBOCRRecordLayout, ParsedIBOCR
from instabase.ocr.client.libs.ocr_types import WordPolyDict
from ibformers.datasets.ibds.ibds import get_open_fn
from ibformers.trainer.ib_package.ModelServiceTemplate.src.py.package_name.model import IbModel
from instabase.protos.model_service import model_service_pb2

class InstabaseSDKDummy:
    def __init__(self, file_client: Any, username: str):
        # these will be ignored
        self.file_client = file_client
        self.username = username

    def ibopen(self, path: str, mode: str = 'r') -> Any:
        return open(path, mode)

    def read_file(self, file_path: str) -> str:
        with open(file_path) as f:
            return f.read()

    def write_file(self, file_path: str, content: str):
        with open(file_path, 'w') as f:
            f.write(content)


sdk = InstabaseSDKDummy(None, "user")
open_fn = get_open_fn(sdk)

annotation_file = Path(__file__).parent.parent / "ibformers" / "example" / "UberEats.ibannotator"
with open(str(annotation_file), 'r') as fl:
    content = json.load(fl)


# Load model
model_path = '/var/folders/k_/kwwcpz8944j2c5xbyfpypmgc0000gn/T/tmpytuo3hie/model'
model = IbModel(model_data_path=model_path)
model.load()


for annotations in content['files']:
    ocr_path = annotations['ocrPath']
    req = model_service_pb2.RunModelRequest(input_path=ocr_path)
    out = model.run(req)







