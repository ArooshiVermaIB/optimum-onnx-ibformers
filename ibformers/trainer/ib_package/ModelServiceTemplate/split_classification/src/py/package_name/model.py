import json
import json
import os

from ibformers.data.collators.collmenter import CollatorWithAugmentation
from ibformers.data.utils import convert_to_dict_of_lists
from ibformers.datasets.ib_common import IB_DATASETS

os.environ["HF_DATASETS_OFFLINE"] = "1"
import sys

# TODO: remove once packages paths defined in extraction.json will be added to PYTHONPATH

from ibformers.trainer.arguments import EnhancedTrainingArguments
from ibformers.trainer.ib_utils import InstabaseSDK

pth, _ = os.path.split(__file__)
if pth not in sys.path:
    sys.path.append(pth)


import tempfile
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from instabase.model_service.input_utils import resolve_parsed_ibocr_from_request
from instabase.model_service.model_cache import Model
from instabase.protos.model_service import model_service_pb2
from instabase.model_service.file_utils import get_file_client
from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedModel, AutoConfig
from ibformers.data.pipelines.pipeline import PIPELINES, prepare_dataset
from ibformers.datasets import DATASETS_PATH
from ibformers.trainer.trainer import IbTrainer

MODEL_PATH = Path(__file__).parent / "model_data"


class IbModel(Model):
    """Class for handling inference of models trained by ibformers library"""

    def __init__(self, model_data_path: str = None, ibsdk: InstabaseSDK = None) -> None:
        if model_data_path is None:
            self.model_data_path = MODEL_PATH
        else:
            self.model_data_path = model_data_path
        self.tokenizer: Optional[PreTrainedTokenizerFast] = None
        self.model: Optional[PreTrainedModel] = None
        self.device: Optional[str] = None
        self.pipeline_config = self.load_pipeline_config(self.model_data_path)
        self.pipeline = PIPELINES[self.pipeline_config["pipeline_name"]]
        # add file client in case we would need to donwload images from instabase file system
        self.file_client = get_file_client()

    def get_ibsdk(self, username):
        return InstabaseSDK(file_client=self.file_client, username=username)

    @staticmethod
    def load_pipeline_config(path):
        with open(os.path.join(path, "pipeline.json"), "r") as f:
            config = json.load(f)
        return config

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_data_path, use_fast=True)
        model_class = self.pipeline["model_class"]

        # Get the config class for the model and initialize it from the config file
        config_class = getattr(model_class, "config_class", AutoConfig)
        config = config_class.from_pretrained(self.model_data_path)

        self.model = model_class.from_pretrained(self.model_data_path, config=config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        compute_metrics = self.pipeline["compute_metrics"]

        # Data collator
        data_collator = CollatorWithAugmentation(tokenizer=self.tokenizer, pad_to_multiple_of=8, model=self.model)

        # Get training arguments
        train_args = torch.load(os.path.join(self.model_data_path, "training_args.bin"))

        # Initialize our Trainer, trainer class will be used only for prediction
        self.trainer = IbTrainer(
            args=EnhancedTrainingArguments(
                output_dir=tempfile.TemporaryDirectory().name,
                per_device_eval_batch_size=2,
                report_to="none",
                label_names=train_args.label_names,
            ),
            model=self.model,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            post_process_function=None,
        )

    def cleanup(self) -> None:
        self.tokenizer = None
        self.model = None
        self.device = None
        torch.cuda.empty_cache()

    def run(self, request: model_service_pb2.RunModelRequest) -> model_service_pb2.ModelResult:
        assert (
            self.tokenizer is not None and self.model is not None
        ), "Trying to run a model that has not yet been loaded"
        parsed_ibocr = resolve_parsed_ibocr_from_request(request)
        ds_path = Path(DATASETS_PATH) / self.pipeline_config["dataset_name_or_path"]
        # prepare config kwargs
        load_kwargs = self.pipeline["dataset_load_kwargs"]

        ibsdk = self.get_ibsdk(request.context.username)
        load_kwargs["ibsdk"] = ibsdk
        if hasattr(self.model.config, "ib_id2label"):
            id2label = dict((int(key), value) for key, value in self.model.config.ib_id2label.items())
            load_kwargs["id2label"] = id2label

        # get proper dataset class and config
        dataset_class = IB_DATASETS[self.pipeline_config["dataset_name_or_path"]]
        config_class = dataset_class.BUILDER_CONFIG_CLASS
        config = config_class(name=self.pipeline_config["dataset_name_or_path"], version="1.0.0", **load_kwargs)
        image_processor = dataset_class.create_image_processor(config)

        # process the data
        examples = dataset_class.get_examples_from_model_service([parsed_ibocr], config, image_processor)

        if len(examples) == 0:
            # See assert_valid_record for what makes a record valid.
            raise ValueError(
                "Found no records to process for inference. Check if input documents have records with valid OCR output."
            )

        prediction_schema = dataset_class.get_inference_dataset_features(config)
        data = convert_to_dict_of_lists(examples, examples[0].keys())
        predict_dataset = Dataset.from_dict(data, prediction_schema)

        fn_kwargs = {**self.pipeline_config, **{"tokenizer": self.tokenizer}}

        map_kwargs = {
            "num_proc": 1,
            "load_from_cache_file": False,
            "keep_in_memory": True,
            "fn_kwargs": {"split_name": "predict", **fn_kwargs},
        }

        processed_dataset = prepare_dataset(predict_dataset, self.pipeline, **map_kwargs)
        prediction_output = self.trainer.predict(processed_dataset)
        predictions = prediction_output.metrics["test_predictions"]

        assert len(predictions) == 1, "Multiple ibdocs given, please check"
        split_preds = predictions[next(iter(predictions))]["prediction"]
        pred_json = json.dumps(split_preds)
        raw_data = model_service_pb2.RawData(type="str", data=pred_json.encode("utf-8"))
        return model_service_pb2.ModelResult(raw_data=raw_data)
