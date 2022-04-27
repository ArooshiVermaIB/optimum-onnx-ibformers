import logging
from functools import partial

from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
)

from ibformers.data.chunk import produce_chunks, pairer
from ibformers.data.collators.collators.composite import TableDetrCollator
from ibformers.data.metrics import (
    compute_legacy_metrics_for_sl,
    compute_legacy_metrics_for_mqa,
    compute_metrics_for_qa_task,
    compute_metrics_for_singleqa_task,
    compute_metrics_for_sc,
    compute_metrics_for_cls,
)
from ibformers.data.predict_splinter import squad_metric, splinter_metric, compute_metrics_for_splinter_mqa
from ibformers.data.predict_table import get_predictions_for_table_detr
from ibformers.data.splinter_processing import find_recurring_spans, build_prefix_with_mqa_splinter
from ibformers.data.tokenize import tokenize, tokenize_layoutlmv2
from ibformers.data.transform import (
    norm_bboxes_for_layoutlm,
    build_prefix_with_mqa_ids,
    fuzzy_tag_in_document,
    build_prefix_single_qa,
    token_spans_to_start_end,
    convert_from_mrqa_fmt,
    prepare_input_squad,
)
from ibformers.data.transform.stack_pages import filter_npages
from ibformers.data.transform.table import (
    prepare_image_and_table_data,
    calculate_margins,
    normalize_object_bboxes,
    convert_bboxes_to_center_based,
)
from ibformers.models.automodel_utils import AutoModelForSplitClassification
from ibformers.models.bbox_masking_models import LayoutLMForMaskedLMAndLayout, LayoutLMForMaskedLMAndLayoutRegression
from ibformers.models.layoutlm_positionless import (
    LayoutLMPositionlessForMaskedLMAndLayoutRegression,
)
from ibformers.models.layv1mqa import LayMQAForTokenClassification
from ibformers.models.layv1splinter import LayoutSplinterModel
from ibformers.models.table_detr import CombinedTableDetrModel
from ibformers.trainer.table_trainer import TableIbTrainer


def chain(example_batch, fn_lst, **kwargs):
    for fn in fn_lst:
        example_batch = fn(example_batch, **kwargs)
    return example_batch


def pipeline_preprocess(dataset, fn_lst, chain_functions=True, fn_kwargs=None, batch_size=4, num_proc=4, **map_kwargs):
    """
    :param dataset: hf/dataset used for preprocessing
    :param fn_lst: list of functions to apply (via map method) to the dataset
    :param chain_functions: whether to chain functions to get rid of caching step after each function
    :param fn_kwargs: kwargs passed to the functions in the fn_lst
    :param batch_size: size of the batch
    :param num_proc: number of processes to use for fn mapping
    :param map_kwargs: additional parameters passed to map method
    :return: preprocessed hf/dataset
    """
    if fn_kwargs is None:
        fn_kwargs = {}
    ds = dataset
    if chain_functions:
        fn_lst = [partial(chain, fn_lst=fn_lst)]

    for fn_idx, fn in enumerate(fn_lst):
        logging.debug(f"apply function number {fn_idx} in the pipeline")
        ds: Dataset = ds.map(
            fn,
            batched=True,
            batch_size=batch_size,
            fn_kwargs=fn_kwargs,
            num_proc=num_proc,
            **map_kwargs,
        )
    return ds


def map_column_names(dataset, column_mapping):
    ds = dataset
    for col_old, col_new in column_mapping:
        ds = ds.rename_column(col_old, col_new)
    return ds


def prepare_dataset(dataset, pipeline, **kwargs):
    # preprocess
    map_fn_lst = pipeline["preprocess"]
    preprocess_kwargs = pipeline.get("preprocess_kwargs", {})
    # add pipeline specific arguments. Keys passed in kwargs will overwrite pipeline specific kwargs
    for k, v in preprocess_kwargs.items():
        if k in kwargs["fn_kwargs"]:
            logging.warning(f'{k}={v} will be overwritetten by {kwargs["fn_kwargs"][k]}')
        else:
            kwargs["fn_kwargs"][k] = v
    dataset = pipeline_preprocess(dataset, fn_lst=map_fn_lst, **kwargs)
    # rename
    column_mapping = pipeline["column_mapping"]
    dataset = map_column_names(dataset, column_mapping=column_mapping)

    return dataset


# layoutlm sequence labeling pipeline
layoutlm_sl = {
    "dataset_load_kwargs": {},
    "preprocess": [tokenize, norm_bboxes_for_layoutlm, produce_chunks],
    "preprocess_kwargs": {"chunking_strategy": "SINGLE_PAGES"},
    "column_mapping": [("token_label_ids", "labels"), ("bboxes", "bbox")],
    "augmenters_kwargs": {"augmenters_list": ["bbox"]},
    "model_class": AutoModelForTokenClassification,
    "compute_metrics": compute_legacy_metrics_for_sl,
}

layoutxlm_sl = {
    "dataset_load_kwargs": {"use_image": True},
    "preprocess": [tokenize, norm_bboxes_for_layoutlm, produce_chunks],
    "preprocess_kwargs": {"chunking_strategy": "SINGLE_PAGES"},
    "column_mapping": [("token_label_ids", "labels"), ("bboxes", "bbox"), ("images", "image")],
    "augmenters_kwargs": {"augmenters_list": ["bbox"]},
    "model_class": AutoModelForTokenClassification,
    "compute_metrics": compute_legacy_metrics_for_sl,
}

layoutlmv2_sl = {
    "dataset_load_kwargs": {"use_image": True},
    "preprocess": [tokenize_layoutlmv2, norm_bboxes_for_layoutlm, produce_chunks],
    "preprocess_kwargs": {"chunking_strategy": "SINGLE_PAGES"},
    "column_mapping": [("token_label_ids", "labels"), ("bboxes", "bbox"), ("images", "image")],
    "augmenters_kwargs": {"augmenters_list": ["bbox"]},
    "model_class": AutoModelForTokenClassification,
    "compute_metrics": compute_legacy_metrics_for_sl,
}

laymqav1 = {
    "dataset_load_kwargs": {"use_image": False},
    "preprocess": [build_prefix_with_mqa_ids, tokenize, norm_bboxes_for_layoutlm, produce_chunks],
    "column_mapping": [("token_label_ids", "labels"), ("bboxes", "bbox")],
    "augmenters_kwargs": {"augmenters_list": []},
    "model_class": LayMQAForTokenClassification,
    "compute_metrics": compute_legacy_metrics_for_mqa,
}


from_docvqa_to_mqa = {
    "dataset_load_kwargs": {"use_image": False},
    "preprocess": [
        fuzzy_tag_in_document,
        build_prefix_with_mqa_ids,
        tokenize,
        norm_bboxes_for_layoutlm,
        produce_chunks,
    ],
    "column_mapping": [("token_label_ids", "labels"), ("bboxes", "bbox")],
    "augmenters_kwargs": {"augmenters_list": []},
    "model_class": LayMQAForTokenClassification,
    "compute_metrics": compute_metrics_for_qa_task,
}

from_websrc_to_mqa = {
    "dataset_load_kwargs": {"use_image": False},
    "preprocess": [
        build_prefix_with_mqa_ids,
        tokenize,
        norm_bboxes_for_layoutlm,
        produce_chunks,
    ],
    "preprocess_kwargs": {"convert_to_question": False, "shuffle_mqa_ids": True},
    "column_mapping": [("token_label_ids", "labels"), ("bboxes", "bbox")],
    "augmenters_kwargs": {"augmenters_list": ["bbox"]},
    "model_class": LayMQAForTokenClassification,
    "compute_metrics": compute_metrics_for_qa_task,
}

plain_sl = {
    "dataset_load_kwargs": {},
    "preprocess": [tokenize, produce_chunks],
    "column_mapping": [("token_label_ids", "labels")],
    "augmenters_kwargs": {"augmenters_list": []},
    "model_class": AutoModelForTokenClassification,
    "compute_metrics": compute_legacy_metrics_for_sl,
}

single_qa = {
    "dataset_load_kwargs": {"use_image": False},
    "preprocess": [build_prefix_single_qa, tokenize, produce_chunks, token_spans_to_start_end],
    "preprocess_kwargs": {"save_memory": False},  # token_spans_to_start_end is after chunking and it requires entities
    "column_mapping": [("token_label_ids", "labels")],
    "augmenters_kwargs": {"augmenters_list": []},
    "model_class": AutoModelForQuestionAnswering,
    "compute_metrics": compute_metrics_for_singleqa_task,
}

# image table extraction
table_transformer = {
    "dataset_load_kwargs": {"use_image": True},
    "preprocess": [
        calculate_margins,
        prepare_image_and_table_data,
        normalize_object_bboxes,
        convert_bboxes_to_center_based,
    ],
    "augmenters_kwargs": {"augmenters_list": []},
    "custom_collate": TableDetrCollator,
    "column_mapping": [],
    "model_class": CombinedTableDetrModel,
    "needs_tokenizer": False,
    "compute_metrics": get_predictions_for_table_detr,
    "trainer_class": TableIbTrainer,
}


# mlm pretraining
layoutlm_mlm = {
    "dataset_load_kwargs": {},
    "preprocess": [tokenize, norm_bboxes_for_layoutlm, produce_chunks],
    "column_mapping": [("bboxes", "bbox")],
    "augmenters_kwargs": {"augmenters_list": ["mlm"]},
    "model_class": AutoModelForMaskedLM,
    "compute_metrics": None,
}

layoutlm_mlm_bm = {
    "dataset_load_kwargs": {},
    "preprocess": [tokenize, norm_bboxes_for_layoutlm, produce_chunks],
    "column_mapping": [("bboxes", "bbox")],
    "augmenters_kwargs": {"augmenters_list": ["mlm", "bbox_masking"]},
    "model_class": LayoutLMForMaskedLMAndLayout,
    "compute_metrics": None,
}

layoutlm_mlm_bm_regresssion = {
    "dataset_load_kwargs": {},
    "preprocess": [tokenize, norm_bboxes_for_layoutlm, produce_chunks],
    "column_mapping": [("bboxes", "bbox")],
    "augmenters_kwargs": {"augmenters_list": ["mlm", "bbox_masking"]},
    "model_class": LayoutLMForMaskedLMAndLayoutRegression,
    "compute_metrics": None,
}


layoutlm_mlm_bm_regresssion_positionless = {
    "dataset_load_kwargs": {},
    "preprocess": [tokenize, norm_bboxes_for_layoutlm, produce_chunks],
    "column_mapping": [("bboxes", "bbox")],
    "augmenters_kwargs": {"augmenters_list": ["mlm", "bbox_masking"]},
    "model_class": LayoutLMPositionlessForMaskedLMAndLayoutRegression,
    "compute_metrics": None,
}


plain_mlm = {
    "dataset_load_kwargs": {},
    "preprocess": [tokenize, produce_chunks],
    "column_mapping": [("token_label_ids", "labels")],
    "augmenters_kwargs": {"augmenters_list": ["mlm"]},
    "model_class": AutoModelForMaskedLM,
    "compute_metrics": compute_legacy_metrics_for_sl,
}

# original splinter with QA head
splinter_qa = {
    "dataset_load_kwargs": {},
    "preprocess": [
        convert_from_mrqa_fmt,
        fuzzy_tag_in_document,
        build_prefix_with_mqa_splinter,
        tokenize,
        produce_chunks,
        token_spans_to_start_end,
    ],
    "column_mapping": [],
    "augmenters_kwargs": {"augmenters_list": []},
    "model_class": AutoModelForQuestionAnswering,
    "compute_metrics": squad_metric,
}

# Training QA models on public datasets (docvqa, squad)
squad_qa = {
    "dataset_load_kwargs": {},
    "preprocess": [
        convert_from_mrqa_fmt,
        fuzzy_tag_in_document,
        prepare_input_squad,
        tokenize,
        produce_chunks,
        token_spans_to_start_end,
    ],
    "column_mapping": [],
    "augmenters_kwargs": {"augmenters_list": []},
    "model_class": AutoModelForQuestionAnswering,
    "compute_metrics": squad_metric,
}

# pipeline for Splinter unsupervised training
splinter_unsupervised = {
    "dataset_load_kwargs": {},
    "preprocess": [
        tokenize,
        produce_chunks,
        find_recurring_spans,
    ],
    "column_mapping": [("qid", "id")],
    "augmenters_kwargs": {"augmenters_list": []},
    "model_class": LayoutSplinterModel,
    "compute_metrics": splinter_metric,
}

# to use in the extraction datasets (e.g. benchmarks)
splinter_sl = {
    "dataset_load_kwargs": {"use_image": False},
    "preprocess": [build_prefix_with_mqa_splinter, tokenize, norm_bboxes_for_layoutlm, produce_chunks],
    "column_mapping": [("token_label_ids", "labels"), ("bboxes", "bbox")],
    "augmenters_kwargs": {"augmenters_list": []},
    "model_class": LayoutSplinterModel,
    "compute_metrics": compute_metrics_for_splinter_mqa,
}

# pretrain splinter in the qa dataset (docvqa, squad)
docvqa_splinter_sl = {
    "dataset_load_kwargs": {},
    "preprocess": [
        convert_from_mrqa_fmt,
        fuzzy_tag_in_document,
        build_prefix_with_mqa_splinter,
        tokenize,
        produce_chunks,
    ],
    "column_mapping": [],
    "augmenters_kwargs": {"augmenters_list": []},
    "model_class": LayoutSplinterModel,
    "compute_metrics": splinter_metric,
}

layoutlm_sc = {
    "dataset_load_kwargs": {},
    "preprocess": [tokenize, norm_bboxes_for_layoutlm, pairer],
    "column_mapping": [],
    "augmenters_kwargs": {"augmenters_list": []},
    "model_class": AutoModelForSplitClassification,
    "compute_metrics": compute_metrics_for_sc,
}

plain_text_cls = {
    "dataset_load_kwargs": {},
    # tokenizer is passed to pipeline externally: in train.run_train tokenizer is passed to map_kwargs,
    # which itself is a kwarg of prepare_dataset and pipeline_preprocess
    "preprocess": [filter_npages, tokenize, produce_chunks],
    "column_mapping": [("class_label", "labels")],
    "augmenters_kwargs": {"augmenters_list": []},
    "model_class": AutoModelForSequenceClassification,
    "compute_metrics": compute_metrics_for_cls,
}

layoutlm_cls = {
    "dataset_load_kwargs": {},
    # tokenizer is passed to pipeline externally: in train.run_train tokenizer is passed to map_kwargs,
    # which itself is a kwarg of prepare_dataset and pipeline_preprocess
    "preprocess": [filter_npages, tokenize, norm_bboxes_for_layoutlm, produce_chunks],
    "column_mapping": [("class_label", "labels"), ("bboxes", "bbox")],
    "augmenters_kwargs": {"augmenters_list": []},
    "model_class": AutoModelForSequenceClassification,
    "compute_metrics": compute_metrics_for_cls,
}


PIPELINES = {
    "layoutlm_sl": layoutlm_sl,
    "layoutlmv2_sl": layoutlmv2_sl,
    "layoutxlm_sl": layoutxlm_sl,
    "laymqav1": laymqav1,
    "from_docvqa_to_mqa": from_docvqa_to_mqa,
    "plain_sl": plain_sl,
    "plain_mlm": plain_mlm,
    "layoutlm_mlm": layoutlm_mlm,
    "single_qa": single_qa,
    "from_websrc_to_mqa": from_websrc_to_mqa,
    "layoutlm_mlm_bm": layoutlm_mlm_bm,
    "layoutlm_mlm_bm_regresssion": layoutlm_mlm_bm_regresssion,
    "splinter_qa": splinter_qa,
    "squad_qa": squad_qa,
    "splinter_unsupervised": splinter_unsupervised,
    "splinter_sl": splinter_sl,
    "docvqa_splinter_sl": docvqa_splinter_sl,
    "layoutlm_mlm_bm_regresssion_positionless": layoutlm_mlm_bm_regresssion_positionless,
    "layoutlm_sc": layoutlm_sc,
    "plain_text_cls": plain_text_cls,
    "layoutlm_cls": layoutlm_cls,
    "table_transformer": table_transformer,
}
