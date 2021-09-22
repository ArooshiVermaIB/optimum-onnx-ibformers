import json
import os
from pathlib import Path
from typing import List, Tuple, Mapping, Optional, NamedTuple

import torch
from datasets import load_dataset, GenerateMode
from instabase.model_service.input_utils import resolve_parsed_ibocr_from_request
from instabase.model_service.model_cache import Model
from instabase.ocr.client.libs.algorithms import WordPolyInputColMapper
from instabase.protos.model_service import model_service_pb2
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizerFast, AutoTokenizer
from ibformers.data.pipelines.pipeline import PIPELINES, prepare_dataset
from ibformers.datasets import DATASETS_PATH
from ibformers.trainer.ib_utils import _abspath
from ibformers.trainer.trainer import IbTrainer


MODEL_PATH = _abspath('model_data/')


class IbModel(Model):
    """Class for handling inference of models trained by ibformers library"""

    def __init__(self, model_data_path: str = None) -> None:
        if model_data_path is None:
            self.model_data_path = MODEL_PATH
        else:
            self.model_data_path = model_data_path
        self.tokenizer: Optional[LayoutLMTokenizerFast] = None
        self.model: Optional[LayoutLMForTokenClassification] = None
        self.device: Optional[str] = None
        self.pipeline_config = self.load_pipeline_config(model_data_path)
        self.pipeline = PIPELINES[self.pipeline_config['pipeline_name']]

    def load_pipeline_config(self, path):
        with open(os.path.join(path, 'pipeline.json'), 'r') as f:
            config = json.load(f)
        return config

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_data_path, use_fast=True)
        model_class = self.pipeline['model_class']

        self.model = model_class.from_pretrained(self.model_data_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        collate_fn = self.pipeline["collate"]
        compute_metrics = self.pipeline['compute_metrics']

        # Data collator
        data_collator = collate_fn(self.tokenizer, pad_to_multiple_of=8, model=self.model)

        # Initialize our Trainer, trainer class will be used only for prediction
        self.trainer = IbTrainer(
            model=self.model,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            post_process_function=None
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

        # TODO: investigate if we can remove outputting input column mapper indexes,
        #  it requires lots of additional computation - two additional iteration over all words in document
        record_joined, err = parsed_ibocr.get_joined_page()
        if err:
            raise RuntimeError(f"Error while getting parsed_ibocr.get_joined_page(): {err}")
        mapper = WordPolyInputColMapper(record_joined)  # not good
        word_polys: List['WordPolyDict'] = []
        layouts = []
        for record in parsed_ibocr.get_ibocr_records():
            word_polys += [i for j in record.get_lines() for i in j]
            l = record.get_metadata_list()
            layouts.extend([i.get_layout() for i in l])

        # pass single document and create in memory dataset
        ds_path = Path(DATASETS_PATH) / self.pipeline_config['dataset_name_or_path']
        name_to_use = str(ds_path) if ds_path.is_dir() else self.pipeline_config['dataset_name_or_path']
        load_kwargs = self.pipeline["dataset_load_kwargs"]
        if hasattr(self.model.config, 'id2label'):
            load_kwargs['id2label'] = self.model.config.id2label

        predict_dataset = load_dataset(
            path=name_to_use,
            name=self.pipeline_config['dataset_config_name'],
            data_files={'test': [parsed_ibocr]},
            ignore_verifications=True,
            keep_in_memory=True,
            split='test',
            download_mode=GenerateMode.FORCE_REDOWNLOAD,
            **load_kwargs,
        )

        fn_kwargs = {**self.pipeline_config, **{'tokenizer': self.tokenizer}}

        map_kwargs = {'num_proc': 1,
                      'load_from_cache_file': False,
                      'keep_in_memory': True,
                      'fn_kwargs': fn_kwargs
                      }

        processed_dataset = prepare_dataset(predict_dataset, self.pipeline, **map_kwargs)
        prediction_output = self.trainer.predict(processed_dataset)
        predictions = prediction_output.metrics['test_predictions']

        assert len(predictions) == 1, 'Should output predictions only for one document'
        doc_pred = list(predictions.values())[0]

        entities = []
        for field, field_pred in doc_pred.items():
            for word in field_pred['words']:
                token_idx = word['idx']
                original_word_poly = word_polys[token_idx]
                start = mapper.get_index(original_word_poly)
                assert word['word'] == original_word_poly['word']
                ner_result = model_service_pb2.NERTokenResult(
                    content=word['word'],
                    label=field,
                    score=word['conf'],
                    start_index=start,
                    end_index=start + len(word['word']),
                )
                entities.append(ner_result)
        return model_service_pb2.ModelResult(
            ner_result=model_service_pb2.NERResult(entities=entities)
        )
