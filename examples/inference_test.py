import json
from pathlib import Path
from ibformers.trainer.ib_package.ModelServiceTemplate.src.py.package_name.model import IbModel
from instabase.protos.model_service import model_service_pb2

annotation_file = Path("/Users/rafalpowalski/python/annotation/uber/UberEats.ibannotator")
with open(str(annotation_file), 'r') as fl:
    content = json.load(fl)


# Load model
model_path = '/Users/rafalpowalski/python/models/test_model/artifact/src/py/CustomModel/model_data'
model = IbModel(model_data_path=model_path)
model.load()

outs = []
for annotations in content['files']:
    ocr_path = annotations['ocrPath']
    req = model_service_pb2.RunModelRequest(input_path=ocr_path)
    out = model.run(req)
    outs.append(out)


print(len(outs))
