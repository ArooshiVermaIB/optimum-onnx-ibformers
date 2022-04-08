"""
Copyright (C) 2021 Microsoft Corporation
"""
import statistics as stat
from collections import Counter
from difflib import SequenceMatcher

import numpy as np

import ibformers.third_party.grits.eval_utils as eval_utils


def transpose(matrix):
    return list(map(list, zip(*matrix)))


def align_1d(sequence1, sequence2, reward_function, return_alignment=False):
    """
    Dynamic programming sequence alignment between two sequences
    Traceback convention: -1 = up, 1 = left, 0 = diag up-left
    """
    sequence1_length = len(sequence1)
    sequence2_length = len(sequence2)

    scores = np.zeros((sequence1_length + 1, sequence2_length + 1))
    pointers = np.zeros((sequence1_length + 1, sequence2_length + 1))

    # Initialize first column
    for row_idx in range(1, sequence1_length + 1):
        pointers[row_idx, 0] = -1

    # Initialize first row
    for col_idx in range(1, sequence2_length + 1):
        pointers[0, col_idx] = 1

    for row_idx in range(1, sequence1_length + 1):
        for col_idx in range(1, sequence2_length + 1):
            reward = reward_function(sequence1[row_idx - 1], sequence2[col_idx - 1])
            diag_score = scores[row_idx - 1, col_idx - 1] + reward
            same_row_score = scores[row_idx, col_idx - 1]
            same_col_score = scores[row_idx - 1, col_idx]

            max_score = max(diag_score, same_col_score, same_row_score)
            scores[row_idx, col_idx] = max_score
            if diag_score == max_score:
                pointers[row_idx, col_idx] = 0
            elif same_col_score == max_score:
                pointers[row_idx, col_idx] = -1
            else:
                pointers[row_idx, col_idx] = 1

    score = scores[sequence1_length, sequence2_length]
    score = 2 * score / (sequence1_length + sequence2_length)

    if not return_alignment:
        return score

    # Backtrace
    cur_row = sequence1_length
    cur_col = sequence2_length
    aligned_sequence1_indices = []
    aligned_sequence2_indices = []
    while not (cur_row == 0 and cur_col == 0):
        if pointers[cur_row, cur_col] == -1:
            cur_row -= 1
        elif pointers[cur_row, cur_col] == 1:
            cur_col -= 1
        else:
            cur_row -= 1
            cur_col -= 1
            aligned_sequence1_indices.append(cur_col)
            aligned_sequence2_indices.append(cur_row)

    aligned_sequence1_indices = aligned_sequence1_indices[::-1]
    aligned_sequence2_indices = aligned_sequence2_indices[::-1]

    return aligned_sequence1_indices, aligned_sequence2_indices, score


def objects_to_cells(
    bboxes, labels, scores, page_tokens, structure_class_names, structure_class_thresholds, structure_class_map
):
    bboxes, scores, labels = eval_utils.apply_class_thresholds(
        bboxes, labels, scores, structure_class_names, structure_class_thresholds
    )

    table_objects = []
    for bbox, score, label in zip(bboxes, scores, labels):
        table_objects.append({"bbox": bbox, "score": score, "label": label})

    table = {"objects": table_objects, "page_num": 0}

    table_class_objects = [obj for obj in table_objects if obj["label"] == structure_class_map["table"]]
    if len(table_class_objects) > 1:
        table_class_objects = sorted(table_class_objects, key=lambda x: x["score"], reverse=True)
    try:
        table_bbox = list(table_class_objects[0]["bbox"])
    except:
        table_bbox = (0, 0, 1000, 1000)

    tokens_in_table = [token for token in page_tokens if eval_utils.iob(token["bbox"], table_bbox) >= 0.5]

    # Determine the table cell structure from the objects
    table_structures, cells, confidence_score = eval_utils.objects_to_cells(
        table, table_objects, tokens_in_table, structure_class_names, structure_class_thresholds
    )

    return table_structures, cells, confidence_score, table_bbox


def cells_to_adjacency_pair_list(cells, key="cell_text"):
    # Index the cells by their grid coordinates
    cell_nums_by_coordinates = dict()
    for cell_num, cell in enumerate(cells):
        for row_num in cell["row_nums"]:
            for column_num in cell["column_nums"]:
                cell_nums_by_coordinates[(row_num, column_num)] = cell_num

    # Count the number of unique rows and columns
    row_nums = set()
    column_nums = set()
    for cell in cells:
        for row_num in cell["row_nums"]:
            row_nums.add(row_num)
        for column_num in cell["column_nums"]:
            column_nums.add(column_num)
    num_rows = len(row_nums)
    num_columns = len(column_nums)

    # For each cell, determine its next neighbors
    # - For every row the cell occupies, what is the first cell to the right with text that
    #   also occupies that row
    # - For every column the cell occupies, what is the first cell below with text that
    #   also occupies that column
    adjacency_list = []
    adjacency_bboxes = []
    for cell1_num, cell1 in enumerate(cells):
        # Skip blank cells
        if cell1["cell_text"] == "":
            continue

        adjacent_cell_props = {}
        max_column = max(cell1["column_nums"])
        max_row = max(cell1["row_nums"])

        # For every column the cell occupies...
        for column_num in cell1["column_nums"]:
            # Start from the next row and stop when we encounter a non-blank cell
            # This cell is considered adjacent
            for current_row in range(max_row + 1, num_rows):
                cell2_num = cell_nums_by_coordinates[(current_row, column_num)]
                cell2 = cells[cell2_num]
                if not cell2["cell_text"] == "":
                    adj_bbox = [
                        (max(cell1["bbox"][0], cell2["bbox"][0]) + min(cell1["bbox"][2], cell2["bbox"][2])) / 2 - 3,
                        cell1["bbox"][3],
                        (max(cell1["bbox"][0], cell2["bbox"][0]) + min(cell1["bbox"][2], cell2["bbox"][2])) / 2 + 3,
                        cell2["bbox"][1],
                    ]
                    adjacent_cell_props[cell2_num] = ("V", current_row - max_row - 1, adj_bbox)
                    break

        # For every row the cell occupies...
        for row_num in cell1["row_nums"]:
            # Start from the next column and stop when we encounter a non-blank cell
            # This cell is considered adjacent
            for current_column in range(max_column + 1, num_columns):
                cell2_num = cell_nums_by_coordinates[(row_num, current_column)]
                cell2 = cells[cell2_num]
                if not cell2["cell_text"] == "":
                    adj_bbox = [
                        cell1["bbox"][2],
                        (max(cell1["bbox"][1], cell2["bbox"][1]) + min(cell1["bbox"][3], cell2["bbox"][3])) / 2 - 3,
                        cell2["bbox"][0],
                        (max(cell1["bbox"][1], cell2["bbox"][1]) + min(cell1["bbox"][3], cell2["bbox"][3])) / 2 + 3,
                    ]
                    adjacent_cell_props[cell2_num] = ("H", current_column - max_column - 1, adj_bbox)
                    break

        for adjacent_cell_num, props in adjacent_cell_props.items():
            cell2 = cells[adjacent_cell_num]
            adjacency_list.append((cell1["cell_text"], cell2["cell_text"], props[0], props[1]))
            adjacency_bboxes.append(props[2])

    return adjacency_list, adjacency_bboxes


def cells_to_adjacency_pair_list_with_blanks(cells, key="cell_text"):
    # Index the cells by their grid coordinates
    cell_nums_by_coordinates = dict()
    for cell_num, cell in enumerate(cells):
        for row_num in cell["row_nums"]:
            for column_num in cell["column_nums"]:
                cell_nums_by_coordinates[(row_num, column_num)] = cell_num

    # Count the number of unique rows and columns
    row_nums = set()
    column_nums = set()
    for cell in cells:
        for row_num in cell["row_nums"]:
            row_nums.add(row_num)
        for column_num in cell["column_nums"]:
            column_nums.add(column_num)
    num_rows = len(row_nums)
    num_columns = len(column_nums)

    # For each cell, determine its next neighbors
    # - For every row the cell occupies, what is the next cell to the right
    # - For every column the cell occupies, what is the next cell below
    adjacency_list = []
    adjacency_bboxes = []
    for cell1_num, cell1 in enumerate(cells):
        adjacent_cell_props = {}
        max_column = max(cell1["column_nums"])
        max_row = max(cell1["row_nums"])

        # For every column the cell occupies...
        for column_num in cell1["column_nums"]:
            # The cell in the next row is adjacent
            current_row = max_row + 1
            if current_row >= num_rows:
                continue
            cell2_num = cell_nums_by_coordinates[(current_row, column_num)]
            cell2 = cells[cell2_num]
            adj_bbox = [
                (max(cell1["bbox"][0], cell2["bbox"][0]) + min(cell1["bbox"][2], cell2["bbox"][2])) / 2 - 3,
                cell1["bbox"][3],
                (max(cell1["bbox"][0], cell2["bbox"][0]) + min(cell1["bbox"][2], cell2["bbox"][2])) / 2 + 3,
                cell2["bbox"][1],
            ]
            adjacent_cell_props[cell2_num] = ("V", current_row - max_row - 1, adj_bbox)

        # For every row the cell occupies...
        for row_num in cell1["row_nums"]:
            # The cell in the next column is adjacent
            current_column = max_column + 1
            if current_column >= num_columns:
                continue
            cell2_num = cell_nums_by_coordinates[(row_num, current_column)]
            cell2 = cells[cell2_num]
            adj_bbox = [
                cell1["bbox"][2],
                (max(cell1["bbox"][1], cell2["bbox"][1]) + min(cell1["bbox"][3], cell2["bbox"][3])) / 2 - 3,
                cell2["bbox"][0],
                (max(cell1["bbox"][1], cell2["bbox"][1]) + min(cell1["bbox"][3], cell2["bbox"][3])) / 2 + 3,
            ]
            adjacent_cell_props[cell2_num] = ("H", current_column - max_column - 1, adj_bbox)

        for adjacent_cell_num, props in adjacent_cell_props.items():
            cell2 = cells[adjacent_cell_num]
            adjacency_list.append((cell1["cell_text"], cell2["cell_text"], props[0], props[1]))
            adjacency_bboxes.append(props[2])

    return adjacency_list, adjacency_bboxes


def adjacency_metrics(true_adjacencies, pred_adjacencies):
    true_c = Counter()
    true_c.update([elem for elem in true_adjacencies])

    pred_c = Counter()
    pred_c.update([elem for elem in pred_adjacencies])

    if len(true_adjacencies) > 0:
        recall = (sum(true_c.values()) - sum((true_c - pred_c).values())) / sum(true_c.values())
    else:
        recall = 1
    if len(pred_adjacencies) > 0:
        precision = (sum(pred_c.values()) - sum((pred_c - true_c).values())) / sum(pred_c.values())
    else:
        precision = 1

    if recall + precision == 0:
        f_score = 0
    else:
        f_score = 2 * recall * precision / (recall + precision)

    return recall, precision, f_score


def adjacency_metric(true_cells, pred_cells):
    true_adjacencies, _ = cells_to_adjacency_pair_list(true_cells)
    pred_adjacencies, _ = cells_to_adjacency_pair_list(pred_cells)

    return adjacency_metrics(true_adjacencies, pred_adjacencies)


def adjacency_with_blanks_metric(true_cells, pred_cells):
    true_adjacencies, _ = cells_to_adjacency_pair_list_with_blanks(true_cells)
    pred_adjacencies, _ = cells_to_adjacency_pair_list_with_blanks(pred_cells)

    return adjacency_metrics(true_adjacencies, pred_adjacencies)


def cells_to_grid(cells, key="bbox"):
    if len(cells) == 0:
        return [[]]
    num_rows = max([max(cell["row_nums"]) for cell in cells]) + 1
    num_columns = max([max(cell["column_nums"]) for cell in cells]) + 1
    cell_grid = np.zeros((num_rows, num_columns)).tolist()
    for cell in cells:
        for row_num in cell["row_nums"]:
            for column_num in cell["column_nums"]:
                cell_grid[row_num][column_num] = cell[key]

    return cell_grid


def cells_to_relspan_grid(cells):
    if len(cells) == 0:
        return [[]]
    num_rows = max([max(cell["row_nums"]) for cell in cells]) + 1
    num_columns = max([max(cell["column_nums"]) for cell in cells]) + 1
    cell_grid = np.zeros((num_rows, num_columns)).tolist()
    for cell in cells:
        min_row_num = min(cell["row_nums"])
        min_column_num = min(cell["column_nums"])
        max_row_num = max(cell["row_nums"]) + 1
        max_column_num = max(cell["column_nums"]) + 1
        for row_num in cell["row_nums"]:
            for column_num in cell["column_nums"]:
                cell_grid[row_num][column_num] = [
                    min_column_num - column_num,
                    min_row_num - row_num,
                    max_column_num - column_num,
                    max_row_num - row_num,
                ]

    return cell_grid


def align_cells_outer(true_cells, pred_cells, reward_function):
    """
    Dynamic programming sequence alignment between two sequences
    Traceback convention: -1 = up, 1 = left, 0 = diag up-left
    """

    scores = np.zeros((len(true_cells) + 1, len(pred_cells) + 1))
    pointers = np.zeros((len(true_cells) + 1, len(pred_cells) + 1))

    # Initialize first column
    for row_idx in range(1, len(true_cells) + 1):
        pointers[row_idx, 0] = -1

    # Initialize first row
    for col_idx in range(1, len(pred_cells) + 1):
        pointers[0, col_idx] = 1

    for row_idx in range(1, len(true_cells) + 1):
        for col_idx in range(1, len(pred_cells) + 1):
            reward = align_1d(true_cells[row_idx - 1], pred_cells[col_idx - 1], reward_function)
            diag_score = scores[row_idx - 1, col_idx - 1] + reward
            same_row_score = scores[row_idx, col_idx - 1]
            same_col_score = scores[row_idx - 1, col_idx]

            max_score = max(diag_score, same_col_score, same_row_score)
            scores[row_idx, col_idx] = max_score
            if diag_score == max_score:
                pointers[row_idx, col_idx] = 0
            elif same_col_score == max_score:
                pointers[row_idx, col_idx] = -1
            else:
                pointers[row_idx, col_idx] = 1

    score = scores[len(true_cells), len(pred_cells)]
    if len(pred_cells) > 0:
        precision = score / len(pred_cells)
    else:
        precision = 1
    if len(true_cells) > 0:
        recall = score / len(true_cells)
    else:
        recall = 1
    score = 2 * precision * recall / (precision + recall)
    # score = 2 * score / (len(true_cells) + len(pred_cells))

    cur_row = len(true_cells)
    cur_col = len(pred_cells)
    aligned_true_indices = []
    aligned_pred_indices = []
    while not (cur_row == 0 and cur_col == 0):
        if pointers[cur_row, cur_col] == -1:
            cur_row -= 1
        elif pointers[cur_row, cur_col] == 1:
            cur_col -= 1
        else:
            cur_row -= 1
            cur_col -= 1
            aligned_pred_indices.append(cur_col)
            aligned_true_indices.append(cur_row)

    aligned_true_indices = aligned_true_indices[::-1]
    aligned_pred_indices = aligned_pred_indices[::-1]

    return aligned_true_indices, aligned_pred_indices, score


def factored_2dlcs(true_cell_grid, pred_cell_grid, reward_function):
    true_row_nums, pred_row_nums, row_score = align_cells_outer(true_cell_grid, pred_cell_grid, reward_function)
    true_column_nums, pred_column_nums, column_score = align_cells_outer(
        transpose(true_cell_grid), transpose(pred_cell_grid), reward_function
    )

    score = 0
    for true_row_num, pred_row_num in zip(true_row_nums, pred_row_nums):
        for true_column_num, pred_column_num in zip(true_column_nums, pred_column_nums):
            score += reward_function(
                true_cell_grid[true_row_num][true_column_num], pred_cell_grid[pred_row_num][pred_column_num]
            )

    if true_cell_grid.shape[0] > 0 and true_cell_grid.shape[1] > 0:
        recall = score / (true_cell_grid.shape[0] * true_cell_grid.shape[1])
    else:
        recall = 1
    if pred_cell_grid.shape[0] > 0 and pred_cell_grid.shape[1] > 0:
        precision = score / (pred_cell_grid.shape[0] * pred_cell_grid.shape[1])
    else:
        precision = 1

    if precision > 0 and recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0

    return fscore, precision, recall, row_score, column_score


def lcs_similarity(string1, string2):
    if len(string1) == 0 and len(string2) == 0:
        return 1
    s = SequenceMatcher(None, string1, string2)
    lcs = "".join([string1[block.a : (block.a + block.size)] for block in s.get_matching_blocks()])
    return 2 * len(lcs) / (len(string1) + len(string2))


def compute_metrics(
    true_bboxes, true_labels, true_scores, true_cells, pred_bboxes, pred_labels, pred_scores, pred_cells
):

    # Compute grids/matrices for comparison
    true_relspan_grid = np.array(cells_to_relspan_grid(true_cells))
    true_bbox_grid = np.array(cells_to_grid(true_cells, key="bbox"))
    true_text_grid = np.array(cells_to_grid(true_cells, key="cell_text"), dtype=object)

    pred_relspan_grid = np.array(cells_to_relspan_grid(pred_cells))
    pred_bbox_grid = np.array(cells_to_grid(pred_cells, key="bbox"))
    pred_text_grid = np.array(cells_to_grid(pred_cells, key="cell_text"), dtype=object)

    # ---Compute each of the metrics
    metrics = {}

    # Disabling topology-based metric since it's not easily interpretable
    # (
    #     metrics["grits_top"],
    #     metrics["grits_precision_top"],
    #     metrics["grits_recall_top"],
    #     metrics["grits_top_rowbased"],
    #     metrics["grits_top_columnbased"],
    # ) = factored_2dlcs(true_relspan_grid, pred_relspan_grid, reward_function=eval_utils.iou)

    (
        metrics["grits_loc"],
        metrics["grits_precision_loc"],
        metrics["grits_recall_loc"],
        metrics["grits_loc_rowbased"],
        metrics["grits_loc_columnbased"],
    ) = factored_2dlcs(true_bbox_grid, pred_bbox_grid, reward_function=eval_utils.iou)

    (
        metrics["grits_cont"],
        metrics["grits_precision_cont"],
        metrics["grits_recall_cont"],
        metrics["grits_cont_rowbased"],
        metrics["grits_cont_columnbased"],
    ) = factored_2dlcs(true_text_grid, pred_text_grid, reward_function=lcs_similarity)

    # Disabling adjacency for intepretability reasons
    # (
    #     metrics["adjacency_nonblank_recall"],
    #     metrics["adjacency_nonblank_precision"],
    #     metrics["adjacency_nonblank_fscore"],
    # ) = adjacency_metric(true_cells, pred_cells)
    #
    # (
    #     metrics["adjacency_withblank_recall"],
    #     metrics["adjacency_withblank_precision"],
    #     metrics["adjacency_withblank_fscore"],
    # ) = adjacency_with_blanks_metric(true_cells, pred_cells)

    return metrics


def compute_statistics(structures, cells):
    statistics = {}
    statistics["num_rows"] = len(structures["rows"])
    statistics["num_columns"] = len(structures["columns"])
    statistics["num_cells"] = len(cells)
    statistics["num_spanning_cells"] = len(
        [cell for cell in cells if len(cell["row_nums"]) > 1 or len(cell["column_nums"]) > 1]
    )
    header_rows = set()
    for cell in cells:
        if cell["header"]:
            header_rows = header_rows.union(set(cell["row_nums"]))
    statistics["num_header_rows"] = len(header_rows)
    row_heights = [float(row["bbox"][3] - row["bbox"][1]) for row in structures["rows"]]
    if len(row_heights) >= 2:
        statistics["row_height_coefficient_of_variation"] = stat.stdev(row_heights) / stat.mean(row_heights)
    else:
        statistics["row_height_coefficient_of_variation"] = 0
    column_widths = [float(column["bbox"][2] - column["bbox"][0]) for column in structures["columns"]]
    if len(column_widths) >= 2:
        statistics["column_width_coefficient_of_variation"] = stat.stdev(column_widths) / stat.mean(column_widths)
    else:
        statistics["column_width_coefficient_of_variation"] = 0

    return statistics
