import numpy as np

from ibformers.data.utils import feed_single_example


def _calculate_inclusion_labels(word_bboxes, structure_bboxes):
    labels = np.zeros((word_bboxes.shape[0],), dtype=np.int32)
    sign_flip = np.array([1, 1, -1, -1])
    coord_conditions: np.ndarray = (word_bboxes * sign_flip)[:, None, :] >= (structure_bboxes * sign_flip)[None, :, :]  # type: ignore
    struct_label_mapping = coord_conditions.all(-1).nonzero()
    labels[struct_label_mapping[0]] = struct_label_mapping[1] + 1
    return labels


def _create_labels_for_page(example, page_no):
    word_bboxes = example["bboxes"]
    word_is_in_page = np.array(example["token_page_nums"]) == page_no
    row_labels = np.zeros((word_bboxes.shape[0],), dtype=np.int32)
    col_labels = np.zeros((word_bboxes.shape[0],), dtype=np.int32)
    table_labels = np.zeros((word_bboxes.shape[0],), dtype=np.int32)
    for table_idx in example["tables"]["table_idx"]:
        row_bboxes = np.array(example["tables"]["rows"][table_idx]["bbox"])
        col_bboxes = np.array(example["tables"]["columns"][table_idx]["bbox"])
        table_bboxes = np.array(example["tables"]["table_bboxes"][table_idx])

        row_sublabels = _calculate_inclusion_labels(word_bboxes, row_bboxes)
        col_sublabels = _calculate_inclusion_labels(word_bboxes, col_bboxes)
        table_sublabels = _calculate_inclusion_labels(word_bboxes, table_bboxes)

        row_labels += row_sublabels
        col_labels += col_sublabels
        table_labels += table_sublabels * (table_idx + 1)
    return {
        "token_row_ids": row_labels * word_is_in_page,
        "token_col_ids": col_labels * word_is_in_page,
        "token_table_ids": table_labels * word_is_in_page,
    }


@feed_single_example
def create_non_merged_table_labels(example, **kwargs):
    word_bboxes = example["bboxes"]
    labels = {
        "token_row_ids": np.zeros((word_bboxes.shape[0],), dtype=np.int32),
        "token_col_ids": np.zeros((word_bboxes.shape[0],), dtype=np.int32),
        "token_table_ids": np.zeros((word_bboxes.shape[0],), dtype=np.int32),
    }

    for page_no in np.unique(example["word_page_nums"]):
        page_label_dict = _create_labels_for_page(example, page_no)
        for label_name in labels.keys():
            labels[label_name] += page_label_dict[label_name]
    return labels


@feed_single_example
def stack_table_labels(example, **kwargs):
    stacked_labels = np.stack([example["token_row_ids"], example["token_col_ids"], example["token_table_ids"]], axis=-1)
    return {"stacked_table_labels": stacked_labels}


@feed_single_example
def produce_checkered_ids(example, **kwargs):
    row_ids = np.array(example["token_row_ids"])
    col_ids = np.array(example["token_col_ids"])

    row_mod2 = np.mod(row_ids - 1, 2)
    checkered_row_id = row_mod2 + 1
    checkered_row_id[row_ids == 0] = 0

    col_mod2 = np.mod(col_ids - 1, 2)
    checkered_col_id = col_mod2 + 1
    checkered_col_id[col_ids == 0] = 0

    return {
        "chattered_row_ids": checkered_row_id,
        "chattered_col_ids": checkered_col_id,
    }
