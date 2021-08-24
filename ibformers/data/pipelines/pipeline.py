from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification

from ibformers.data.collators.collate import DataCollatorWithBBoxesForTokenClassification
from ibformers.data.metrics import compute_metrics_for_sl
from ibformers.data.tokenize import tokenize, produce_chunks
from ibformers.data.utils import norm_bboxes_for_layoutlm


def pipeline_preprocess(dataset, fn_lst, batched=True, batch_size=128, **map_kwargs):
    ds = dataset
    for fn in fn_lst:
        ds = ds.map(fn, batched=batched, batch_size=batch_size, **map_kwargs)
    return ds


def map_column_names(dataset, column_mapping):
    ds = dataset
    for col_old, col_new in column_mapping:
        ds = ds.rename_column(col_old, col_new)

    return ds


def prepare_dataset(dataset, pipeline, **map_kwargs):
    # preprocess
    map_fn_lst = pipeline['preprocess']
    dataset = pipeline_preprocess(dataset, fn_lst=map_fn_lst, **map_kwargs)
    # rename
    column_mapping = pipeline['column_mapping']
    dataset = map_column_names(dataset, column_mapping=column_mapping)

    return dataset


# layoutlm sequence labeling pipeline
layoutlm_sl = {'preprocess': [tokenize, norm_bboxes_for_layoutlm, produce_chunks],
               'column_mapping': [('token_label_ids', 'labels'), ('bboxes', 'bbox')],
               'collate': DataCollatorWithBBoxesForTokenClassification,
               'model_class': AutoModelForTokenClassification,
               'compute_metrics': compute_metrics_for_sl}

# TODO: add AutoModel type to pipeline dict

PIPELINES = {'layoutlm_sl': layoutlm_sl}
