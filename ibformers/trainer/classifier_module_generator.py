from instabase.training_utils.model_artifact import ModelArtifactContext

import json
import os
from typing import List


def _make_custom_classifier_script_contents(model_path: str, model_name: str) -> str:
    return f"""
from instabase.model_utils.classifier_connectors import ClassifierModelServiceConnector, make_register_classifiers_fn
from instabase.ocr.client.libs.ibocr import ParsedIBOCR, ParsedIBOCRBuilder

from typing import Tuple
import logging

from google.protobuf import struct_pb2

from instabase.model_utils.classifier import (
    Classifier,
    ClassifierInput,
    ClassifierPrediction,
    ClassifierReportingContext,
    ClassifierTrainingContext,
    DocumentSplitResult,
    DocumentSplitter,
)

from instabase.protos.model_service import model_service_pb2
from instabase.ocr.client.libs.ibocr import ParsedIBOCR


class ModelWrapper(ClassifierModelServiceConnector):
  model_name = '{model_name}'
  model_version = '{{{{published_model_version}}}}'
  force_reload = False
  use_parsed_ibocr = True

  def inner_predict(self,
                    datapoint: ParsedIBOCR) -> Tuple[DocumentSplitResult, str]:
    return None, None

register_classifiers = make_register_classifiers_fn(ModelWrapper)
  """


def write_classifier_module(
    context: ModelArtifactContext, ib_model_path: str, labels: List[str], model_name: str
) -> str:

    classifier_module_path = os.path.join(context.tmp_dir.name, "classifier_mod")
    classifier_path = os.path.join(classifier_module_path, f"{model_name}.ibclassifier")
    classifier_scripts_path = os.path.join(classifier_module_path, "scripts")
    classifier_script_path = os.path.join(classifier_scripts_path, "custom_classifier.py")

    os.makedirs(classifier_module_path, exist_ok=True)
    os.makedirs(classifier_scripts_path, exist_ok=True)

    # Write the custom classifier code
    custom_classifier_script = _make_custom_classifier_script_contents(ib_model_path, model_name)
    with open(classifier_script_path, "w+") as f:
        f.write(custom_classifier_script)

    # Then write the classifier file
    # NOTE: We have some invalid paths below. These paths are not used, but required
    # to make things run...
    classifier_json = {
        "classifier_text": "",
        "scripts_dir": "scripts",
        "type": f"model-service:{model_name}",
        "version": "1.1",
        "model_size": 0,
        "preprocess_settings": None,
        "class_mapping_to_raw_files": {label: [ib_model_path] for label in labels},
        "rootPath": "true",
        "preprocess_output_folder": None,
        "class_mapping_to_training_files": "true",
        "classes": labels,
    }

    with open(classifier_path, "w+") as f:
        f.write(json.dumps(classifier_json))

    return classifier_module_path
