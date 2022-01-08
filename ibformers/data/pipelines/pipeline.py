import logging
from functools import partial

from transformers import AutoModelForTokenClassification, AutoModelForMaskedLM, AutoModelForQuestionAnswering

from ibformers.data.chunk import produce_chunks
from ibformers.data.collators.augmenters.bbox import BboxAugmenter
from ibformers.data.collators.augmenters.bbox_masking import BboxMaskingAugmenter
from ibformers.data.collators.augmenters.mlm import MLMAugmenter
from ibformers.data.collators.collmenter import get_collator_class
from ibformers.data.metrics import (
    compute_legacy_metrics_for_sl,
    compute_legacy_metrics_for_mqa,
    compute_metrics_for_qa_task,
    compute_metrics_for_singleqa_task,
)
from ibformers.data.tokenize import tokenize, tokenize_layoutlmv2
from ibformers.data.transform import (
    norm_bboxes_for_layoutlm,
    build_prefix_with_mqa_ids,
    fuzzy_tag_in_document,
    build_prefix_single_qa,
    token_spans_to_start_end,
)
from ibformers.models.bbox_masking_models import LayoutLMForMaskedLMAndLayout, LayoutLMForMaskedLMAndLayoutRegression
from ibformers.models.layv1mqa import LayMQAForTokenClassification


def chain(example_batch, fn_lst, **kwargs):
    for fn in fn_lst:
        example_batch = fn(example_batch, **kwargs)
    return example_batch


def pipeline_preprocess(
    dataset, fn_lst, chain_functions=True, fn_kwargs=None, batch_size=128, num_proc=4, **map_kwargs
):
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
        ds = ds.map(
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
    "column_mapping": [("token_label_ids", "labels"), ("bboxes", "bbox")],
    "collate": get_collator_class(),
    "model_class": AutoModelForTokenClassification,
    "compute_metrics": compute_legacy_metrics_for_sl,
}

layoutxlm_sl = {
    "dataset_load_kwargs": {"use_image": True},
    "preprocess": [tokenize, norm_bboxes_for_layoutlm, produce_chunks],
    "preprocess_kwargs": {"chunking_strategy": "SINGLE_PAGES"},
    "column_mapping": [("token_label_ids", "labels"), ("bboxes", "bbox"), ("images", "image")],
    "collate": get_collator_class(BboxAugmenter),
    "model_class": AutoModelForTokenClassification,
    "compute_metrics": compute_legacy_metrics_for_sl,
}

layoutlmv2_sl = {
    "dataset_load_kwargs": {"use_image": True},
    "preprocess": [tokenize_layoutlmv2, norm_bboxes_for_layoutlm, produce_chunks],
    "preprocess_kwargs": {"chunking_strategy": "SINGLE_PAGES"},
    "column_mapping": [("token_label_ids", "labels"), ("bboxes", "bbox"), ("images", "image")],
    "collate": get_collator_class(BboxAugmenter),
    "model_class": AutoModelForTokenClassification,
    "compute_metrics": compute_legacy_metrics_for_sl,
}

laymqav1 = {
    "dataset_load_kwargs": {"use_image": False},
    "preprocess": [build_prefix_with_mqa_ids, tokenize, norm_bboxes_for_layoutlm, produce_chunks],
    "column_mapping": [("token_label_ids", "labels"), ("bboxes", "bbox")],
    "collate": get_collator_class(),
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
    "collate": get_collator_class(),
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
    "collate": get_collator_class(BboxAugmenter),
    "model_class": LayMQAForTokenClassification,
    "compute_metrics": compute_metrics_for_qa_task,
}

plain_sl = {
    "dataset_load_kwargs": {},
    "preprocess": [tokenize, produce_chunks],
    "column_mapping": [("token_label_ids", "labels")],
    "collate": get_collator_class(),
    "model_class": AutoModelForTokenClassification,
    "compute_metrics": compute_legacy_metrics_for_sl,
}

single_qa = {
    "dataset_load_kwargs": {"use_image": False},
    "preprocess": [build_prefix_single_qa, tokenize, produce_chunks, token_spans_to_start_end],
    "column_mapping": [("token_label_ids", "labels")],
    "collate": get_collator_class(),
    "model_class": AutoModelForQuestionAnswering,
    "compute_metrics": compute_metrics_for_singleqa_task,
}

# mlm pretraining

layoutlm_mlm = {
    "dataset_load_kwargs": {},
    "preprocess": [tokenize, norm_bboxes_for_layoutlm, produce_chunks],
    "column_mapping": [("bboxes", "bbox")],
    "collate": get_collator_class(MLMAugmenter),
    "model_class": AutoModelForMaskedLM,
    "compute_metrics": None,
}

layoutlm_mlm_bm = {
    "dataset_load_kwargs": {},
    "preprocess": [tokenize, norm_bboxes_for_layoutlm, produce_chunks],
    "column_mapping": [("bboxes", "bbox")],
    "collate": get_collator_class(MLMAugmenter, BboxMaskingAugmenter),
    "model_class": LayoutLMForMaskedLMAndLayout,
    "compute_metrics": None,
}


layoutlm_mlm_bm_regresssion = {
    "dataset_load_kwargs": {},
    "preprocess": [tokenize, norm_bboxes_for_layoutlm, produce_chunks],
    "column_mapping": [("bboxes", "bbox")],
    "collate": get_collator_class(MLMAugmenter, BboxMaskingAugmenter),
    "model_class": LayoutLMForMaskedLMAndLayoutRegression,
    "compute_metrics": None,
}


plain_mlm = {
    "dataset_load_kwargs": {},
    "preprocess": [tokenize, produce_chunks],
    "column_mapping": [("token_label_ids", "labels")],
    "collate": get_collator_class(MLMAugmenter),
    "model_class": AutoModelForMaskedLM,
    "compute_metrics": compute_legacy_metrics_for_sl,
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
}
