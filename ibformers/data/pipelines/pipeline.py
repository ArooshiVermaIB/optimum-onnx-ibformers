from functools import partial

from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification

from ibformers.data.collators.collate import (
    DataCollatorWithBBoxesForTokenClassification,
    DataCollatorWithBBoxesAugmentedForTokenClassification,
    DataCollatorFor1DTokenClassification,
)
from ibformers.data.metrics import (
    compute_metrics_for_sl,
    compute_legacy_metrics_for_sl,
    compute_legacy_metrics_for_mqa,
)
from ibformers.data.tokenize import tokenize, tokenize_layoutlmv2
from ibformers.data.chunk import produce_chunks
from ibformers.data.transform import (
    norm_bboxes_for_layoutlm,
    stack_pages,
    build_prefix_with_special_tokens,
    build_prefix_with_mqa_ids,
    fuzzy_tag_in_document,
)
from ibformers.models.laymqa import LayMQAForSentinelClassification
from ibformers.models.layv1mqa import Layv1MQAForSentinelClassification
from ibformers.models.layv1ttmqa import LayMQAForTokenClassification


def chain(example_batch, fn_lst, **kwargs):
    for fn in fn_lst:
        example_batch = fn(example_batch, **kwargs)
    return example_batch


def pipeline_preprocess(
    dataset, fn_lst, chain_functions=False, fn_kwargs=None, batch_size=128, num_proc=4, **map_kwargs
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

    for fn in fn_lst:
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
    "collate": DataCollatorWithBBoxesForTokenClassification,
    "model_class": AutoModelForTokenClassification,
    "compute_metrics": compute_legacy_metrics_for_sl,
}

layoutxlm_sl = {
    "dataset_load_kwargs": {"use_image": True},
    "preprocess": [tokenize, norm_bboxes_for_layoutlm, produce_chunks, stack_pages],
    "column_mapping": [("token_label_ids", "labels"), ("bboxes", "bbox"), ("images", "image")],
    "collate": DataCollatorWithBBoxesAugmentedForTokenClassification,
    "model_class": AutoModelForTokenClassification,
    "compute_metrics": compute_legacy_metrics_for_sl,
}

layoutlmv2_sl = {
    "dataset_load_kwargs": {"use_image": True},
    "preprocess": [tokenize_layoutlmv2, norm_bboxes_for_layoutlm, produce_chunks, stack_pages],
    "column_mapping": [("token_label_ids", "labels"), ("bboxes", "bbox"), ("images", "image")],
    "collate": DataCollatorWithBBoxesAugmentedForTokenClassification,
    "model_class": AutoModelForTokenClassification,
    "compute_metrics": compute_legacy_metrics_for_sl,
}

layout_mqa = {
    "dataset_load_kwargs": {"use_image": False},
    "preprocess": [
        build_prefix_with_special_tokens,
        tokenize_layoutlmv2,
        norm_bboxes_for_layoutlm,
        produce_chunks,
        stack_pages,
    ],
    "column_mapping": [("token_label_ids", "labels"), ("bboxes", "bbox")],
    "collate": DataCollatorWithBBoxesForTokenClassification,
    "model_class": LayMQAForSentinelClassification,
    "compute_metrics": compute_legacy_metrics_for_mqa,
}

layoutv1_mqa = {
    "dataset_load_kwargs": {"use_image": False},
    "preprocess": [
        build_prefix_with_special_tokens,
        tokenize,
        norm_bboxes_for_layoutlm,
        produce_chunks,
        stack_pages,
    ],
    "column_mapping": [("token_label_ids", "labels"), ("bboxes", "bbox")],
    "collate": DataCollatorWithBBoxesForTokenClassification,
    "model_class": Layv1MQAForSentinelClassification,
    "compute_metrics": compute_legacy_metrics_for_mqa,
}


layoutv1_mqa_emb = {
    "dataset_load_kwargs": {"use_image": False},
    "preprocess": [
        build_prefix_with_mqa_ids,
        tokenize,
        norm_bboxes_for_layoutlm,
        produce_chunks,
        stack_pages,
    ],
    "column_mapping": [("token_label_ids", "labels"), ("bboxes", "bbox")],
    "collate": DataCollatorWithBBoxesForTokenClassification,
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
        stack_pages,
    ],
    "column_mapping": [("token_label_ids", "labels"), ("bboxes", "bbox")],
    "collate": DataCollatorWithBBoxesForTokenClassification,
    "model_class": LayMQAForTokenClassification,
    "compute_metrics": compute_legacy_metrics_for_mqa,
}

plain_sl = {
    "dataset_load_kwargs": {},
    "preprocess": [tokenize, produce_chunks],
    "column_mapping": [("token_label_ids", "labels")],
    "collate": DataCollatorFor1DTokenClassification,
    "model_class": AutoModelForTokenClassification,
    "compute_metrics": compute_legacy_metrics_for_sl,
}

# TODO: add AutoModel type to pipeline dict

PIPELINES = {
    "layoutlm_sl": layoutlm_sl,
    "layoutlmv2_sl": layoutlmv2_sl,
    "layoutxlm_sl": layoutxlm_sl,
    "layout_mqa": layout_mqa,
    "layoutv1_mqa": layoutv1_mqa,
    "layoutv1_mqa_emb": layoutv1_mqa_emb,
    "from_docvqa_to_mqa": from_docvqa_to_mqa,
    "plain_sl": plain_sl,
}
