import json
from pathlib import Path
from ibformers.trainer.ib_package.ModelServiceTemplate.src.py.package_name.model import IbModel
from instabase.protos.model_service import model_service_pb2

annotation_file = Path(__file__).parent.parent / "ibformers" / "example" / "UberEats.ibannotator"
with open(str(annotation_file), 'r') as fl:
    content = json.load(fl)

# Load model
model_path = '/var/folders/k_/kwwcpz8944j2c5xbyfpypmgc0000gn/T/tmpytuo3hie/model'
model = IbModel(model_data_path=model_path)
model.load()

outs = []
for annotations in content['files']:
    ocr_path = annotations['ocrPath']
    req = model_service_pb2.RunModelRequest(input_path=ocr_path)
    out = model.run(req)
    outs.append(out)

print(len(outs))
