import json
from pathlib import Path
from ibformers.trainer.ib_package.ModelServiceTemplate.src.py.package_name.model import IbModel
from instabase.protos.model_service import model_service_pb2

dataset_files = Path("/Users/rafalpowalski/python/annotation/UberEatsDataset/out/s1_process_files")

file_list = dataset_files.glob('*ibdoc')


# Load model
model_path = '/Users/rafalpowalski/python/models/test_model/artifact/src/py/CustomModel/model_data'
model = IbModel(model_data_path=model_path)
model.load()

outs = []
for ocr_path in file_list:
    # with open(ocr_path, "rb") as f:
    #     data = f.read()
    req = model_service_pb2.RunModelRequest(input_path=str(ocr_path))
    out = model.run(req)
    outs.append(out)


print(len(outs))
