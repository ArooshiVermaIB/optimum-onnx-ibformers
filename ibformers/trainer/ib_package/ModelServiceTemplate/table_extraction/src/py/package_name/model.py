import copy
import json
import os
import torch
from ibformers.data.predict_table_ibutils import convert_tables_to_pred_field
from ibformers.data.collators.collmenter import CollatorWithAugmentation
from ibformers.data.utils import convert_to_dict_of_lists
from ibformers.datasets.ib_common import IB_DATASETS
from instabase.dataset_utils.shared_types import (
    PredictionFieldDict,
    IndexedWordDict,
)

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
from typing import List, Optional, Dict, Any, Tuple


from datasets import Dataset
from instabase.model_service.input_utils import resolve_parsed_ibocr_from_request
from instabase.model_service.model_cache import Model
from instabase.ocr.client.libs.algorithms import WordPolyInputColMapper
from instabase.protos.model_service import model_service_pb2
from instabase.model_service.file_utils import get_file_client
from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedModel, PreTrainedTokenizer
from ibformers.data.pipelines.pipeline import PIPELINES, prepare_dataset
from ibformers.trainer.trainer import IbTrainer

MODEL_PATH = Path(__file__).parent / "model_data"


class IbModel(Model):
    """Class for handling inference of models trained by ibformers library"""

    def __init__(self, model_data_path: str = None, ibsdk: InstabaseSDK = None) -> None:
        super().__init__()
        if model_data_path is None:
            self.model_data_path = MODEL_PATH
        else:
            self.model_data_path = model_data_path
        self.eibsdk = ibsdk
        self.tokenizer: Optional[PreTrainedTokenizerFast] = None
        self.model: Optional[PreTrainedModel] = None
        self.device: Optional[str] = None
        self.pipeline_config = self.load_pipeline_config(self.model_data_path)
        self.pipeline = PIPELINES[self.pipeline_config["pipeline_name"]]
        # add file client in case we would need to donwload images from instabase file system
        self.file_client = get_file_client()

    def get_ibsdk(self, username):
        return InstabaseSDK(file_client=self.file_client, username=username) if self.eibsdk is None else self.eibsdk

    @staticmethod
    def load_pipeline_config(path):
        with open(os.path.join(path, "pipeline.json"), "r") as f:
            config = json.load(f)
        return config

    def load(self) -> None:
        if self.pipeline.get("needs_tokenizer", True):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_data_path, use_fast=True)
        else:
            self.tokenizer = PreTrainedTokenizer()

        model_class = self.pipeline["model_class"]

        self.model = model_class.from_pretrained(self.model_data_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        compute_metrics = self.pipeline["compute_metrics"]

        # Data collator
        collator_class = self.pipeline.get("custom_collate", CollatorWithAugmentation)
        data_collator = collator_class(tokenizer=self.tokenizer, pad_to_multiple_of=8, model=self.model)

        # Initialize our Trainer, trainer class will be used only for prediction
        trainer_class = self.pipeline.get("trainer_class", IbTrainer)
        self.trainer = trainer_class(
            args=EnhancedTrainingArguments(
                output_dir=tempfile.TemporaryDirectory().name, per_device_eval_batch_size=2, report_to="none"
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

    def prepare_mapper_and_word_pollys(self, parsed_ibocr):
        # TODO: investigate if we can remove outputting input column mapper indexes,
        #  it requires lots of additional computation - two additional iteration over all words in document
        record_joined, err = copy.deepcopy(parsed_ibocr).get_joined_page()
        if err:
            raise RuntimeError(f"Error while getting parsed_ibocr.get_joined_page(): {err}")
        mapper = WordPolyInputColMapper(record_joined)  # not good
        records = parsed_ibocr.get_ibocr_records()
        if len(records) > 1:
            ValueError("Model should consume only single record. Check if you are passing only single class documents")
        record = records[0]

        word_polys: List["WordPolyDict"] = [i for j in record_joined.get_lines() for i in j]
        return mapper, word_polys, record

    def run(self, request: model_service_pb2.RunModelRequest, ocr_debug_path=None) -> model_service_pb2.ModelResult:
        assert (
            self.tokenizer is not None and self.model is not None
        ), "Trying to run a model that has not yet been loaded"
        parsed_ibocr = resolve_parsed_ibocr_from_request(request)
        mapper, word_polys, record = self.prepare_mapper_and_word_pollys(parsed_ibocr)

        if request.input_path != "":
            record.doc_path = request.input_path

        # prepare config kwargs
        load_kwargs = self.pipeline["dataset_load_kwargs"]

        ibsdk = self.get_ibsdk(request.context.username)
        load_kwargs["ibsdk"] = ibsdk
        if hasattr(self.model.config, "table_id2label"):
            id2label = dict((int(key), value) for key, value in self.model.config.table_id2label.items())
            load_kwargs["table_id2label"] = id2label

        # generate prediction item: (full_path, record_index, record, anno)
        if ocr_debug_path is not None:
            doc_path = ocr_debug_path
        elif request.input_path != "":
            doc_path = request.input_path
        else:
            doc_path = record.get_document_path()

        prediction_item = (doc_path, record.get_ibdoc_record_id(), record, None)

        # get proper dataset class and config
        dataset_class = IB_DATASETS[self.pipeline_config["dataset_name_or_path"]]
        configs = {config.name: config for config in dataset_class.BUILDER_CONFIGS}
        config_class = configs.get(self.pipeline_config["dataset_config_name"])
        for k, v in load_kwargs.items():
            if v is not None:
                if not hasattr(config_class, k):
                    raise ValueError(f"BuilderConfig {config_class} doesn't have a '{k}' key.")
                setattr(config_class, k, v)
        config = config_class
        image_processor = dataset_class.create_image_processor(config)

        # process the data
        examples = dataset_class.get_examples_from_model_service([prediction_item], config, image_processor)

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

        assert len(predictions) == 1, "Should output predictions only for one document"
        doc_pred = list(predictions.values())[0]

        line_indexes = processed_dataset["word_line_idx"][0]
        word_in_line_indexes = processed_dataset["word_in_line_idx"][0]
        indexed_word_mapping = {
            tuple(ind_wrd): glob_idx for glob_idx, ind_wrd in enumerate(zip(line_indexes, word_in_line_indexes))
        }

        fields = []
        record_entities = doc_pred["entities"]
        for field_name, tables in record_entities.items():
            field = convert_tables_to_pred_field(field_name, tables)
            add_index_to_field_inplace(field, indexed_word_mapping, word_polys, mapper)
            fields.append(field)
        # index_spans=get_index_spans(r, indexed_word_mapping, word_polys, mapper),
        pred_json = json.dumps({"fields": fields})
        raw_data = model_service_pb2.RawData(type="Table", data=pred_json.encode("utf-8"))
        output = model_service_pb2.ModelResult(raw_data=raw_data)

        return output


def add_index_to_field_inplace(
    field: PredictionFieldDict, indexed_word_mapping: Dict, word_polys: Any, mapper: Any
) -> PredictionFieldDict:
    for tab in field["table_annotations"]:
        for cell in tab["cells"]:
            ind_spans = get_index_spans(cell["words"], indexed_word_mapping, word_polys, mapper)
            cell["index_spans"] = ind_spans


def get_index_spans(
    ind_words: List[IndexedWordDict], indexed_word_mapping: Dict, word_polys: Any, mapper: Any
) -> List[Tuple[int, int]]:
    index_spans = []
    wrds = []
    for ind_word in ind_words:
        global_w_index = indexed_word_mapping[(ind_word["line_index"], ind_word["word_index"])]
        original_word_poly = word_polys[global_w_index]
        start_index = mapper.get_index(original_word_poly)
        end_index = start_index + len(original_word_poly["word"])
        index_spans.append((start_index, end_index))
        wrds.append(original_word_poly["word"])

    return index_spans
