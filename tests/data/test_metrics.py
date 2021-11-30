import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from ibformers.data import metrics


class TestMetrics(unittest.TestCase):
    def test_iou_score(self):
        # given
        y_true = {"e1": [1, 2, 3]}
        y_pred = {"e1": [1, 2]}
        all_tags = ["e1"]

        # then
        iou_score = metrics.iou_score(y_true, y_pred, all_tags)

        # verify
        expected = {"e1": 2 / 3}
        self.assertDictEqual(iou_score, expected)

    def test_iou_score_empty_lists_and_missing_tags(self):
        # given
        y_true = {"e1": []}
        y_pred = {"e1": []}
        all_tags = ["e1", "e2"]

        # then
        iou_score = metrics.iou_score(y_true, y_pred, all_tags)

        # verify
        expected = {"e1": 0.0, "e2": 0.0}
        self.assertDictEqual(iou_score, expected)

    def test_calculate_average_metrics(self):
        # given
        token_level_df = pd.DataFrame(
            {
                "true_positives": [10, 24, 100, 0, 0],
                "total_positives": [20, 30, 100, 0, 0],
                "total_true": [40, 48, 100, 10, 0],
                "precision": [0.5, 0.8, 1, np.nan, np.nan],
                "recall": [0.25, 0.5, 1, 0, np.nan],
                "f1": [0.333, 0.6153, 1.0, np.nan, np.nan],
            },
            index=["c1", "c2", "c3", "no_predictions", "no_support"],
        )

        # then
        average_metrics = metrics.calculate_average_metrics(token_level_df)

        # verify
        expected_dict = {"micro_precision": 134 / 150, "micro_recall": 134 / 198}
        expected_dict["micro_f1"] = (2 * expected_dict["micro_precision"] * expected_dict["micro_recall"]) / (
            expected_dict["micro_precision"] + expected_dict["micro_recall"]
        )
        expected_dict["macro_precision"] = (0.5 + 0.8 + 1 + 0) / 4
        expected_dict["macro_recall"] = (0.25 + 0.5 + 1 + 0) / 4
        expected_dict["macro_f1"] = (0.333 + 0.6153 + 1 + 0) / 4

        for k, expected_value in expected_dict.items():
            self.assertAlmostEqual(average_metrics[k], expected_value, places=3)

    def test_calculate_average_metrics_all_nan(self):
        # given
        token_level_df = pd.DataFrame(
            {
                "true_positives": [0, 0],
                "total_positives": [0, 0],
                "total_true": [0, 0],
                "precision": [np.nan, np.nan],
                "recall": [np.nan, np.nan],
                "f1": [np.nan, np.nan],
            },
            index=["c1", "c2"],
        )

        # then
        average_metrics = metrics.calculate_average_metrics(token_level_df)

        # verify
        expected_dict = {
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "macro_precision": "NAN",
            "macro_recall": "NAN",
            "macro_f1": "NAN",
        }
        self.assertDictEqual(average_metrics, expected_dict)

    def _get_example_predictions(self):
        return {
            "id_1": {
                "is_test_file": False,
                "entities": {
                    "e1": {
                        "words": [{"idx": 0}, {"idx": 1}, {"idx": 2}, {"idx": 5}],
                        "gold_words": [{"idx": 2}, {"idx": 5}],
                        "text": "t1 t2 t3 t4",
                        "gold_text": "t3 t4",
                        "is_match": False,
                    },
                    "e2": {
                        "words": [{"idx": 3}, {"idx": 4}],
                        "gold_words": [{"idx": 3}, {"idx": 4}],
                        "text": "t5 t6",
                        "gold_text": "t5 t6",
                        "is_match": True,
                    },
                },
            },
            "id_2": {
                "is_test_file": False,
                "entities": {
                    "e1": {
                        "words": [{"idx": 2}, {"idx": 5}],
                        "gold_words": [{"idx": 2}, {"idx": 5}],
                        "text": "t3 t4",
                        "gold_text": "t3 t4",
                        "is_match": True,
                    },
                    "e2": {
                        "words": [{"idx": 4}, {"idx": 5}, {"idx": 6}],
                        "gold_words": [{"idx": 3}, {"idx": 4}],
                        "text": "t5 t6 t7 t8",
                        "gold_text": "t5 t6",
                        "is_match": False,
                    },
                },
            },
        }

    def test_compute_legacy_metrics_for_sl(self):
        # given
        get_predictions_mock = MagicMock()
        get_predictions_mock.return_value = self._get_example_predictions()
        predictions = tuple()
        dataset = None
        label_list = ["O", "e1", "e2"]

        # then
        with patch("ibformers.data.metrics.get_predictions_for_sl", get_predictions_mock):
            calculated_metrics = metrics.compute_legacy_metrics_for_sl(predictions, dataset, label_list)  # type: ignore

        # verify
        expected_metrics = {
            "exact_match": {"e1": 0.5, "e2": 0.5},
            "precision": {"e1": 4 / 6, "e2": 3 / 5},
            "recall": {"e1": 1.0, "e2": 3 / 4},
            "f1": {"e1": 0.8, "e2": 0.6666},
            "micro_precision": 0.636,
            "micro_recall": 0.875,
            "micro_f1": 0.7368,
            "macro_f1": 0.7333,
            "macro_precision": 0.6333,
            "macro_recall": 0.875,
        }
        for metric_name, metric_value in expected_metrics.items():
            if isinstance(metric_value, dict):
                for entity_name, metric_value in metric_value.items():
                    self.assertAlmostEqual(calculated_metrics[metric_name][entity_name], metric_value, places=3)
            else:
                self.assertAlmostEqual(calculated_metrics[metric_name], metric_value, places=3)

    def test_compute_legacy_metrics_for_mqa(self):
        # given
        preds = np.random.random((2, 3, 4))
        labels = [
            [0, 3, 2],
            [1, 3, 2],
        ]
        dataset = MagicMock()
        dataset.__iter__.return_value = [
            {"entities": {"used_label_id": [2, 3, 1]}},
            {"entities": {"used_label_id": [3, 1, 2]}},
        ]
        compute_legacy_metrics_for_sl_mock = MagicMock()

        # then
        with patch("ibformers.data.metrics.compute_legacy_metrics_for_sl", compute_legacy_metrics_for_sl_mock):
            metrics.compute_legacy_metrics_for_mqa((preds, labels), dataset)

        # verify
        (modified_preds, modified_labels), _ = compute_legacy_metrics_for_sl_mock.call_args[0]
        np.testing.assert_array_equal(preds[0][:, [0, 2, 3, 1]], modified_preds[0])
        np.testing.assert_array_equal(preds[1][:, [0, 3, 1, 2]], modified_preds[1])

        np.testing.assert_array_equal([0, 2, 1], modified_labels[0])
        np.testing.assert_array_equal([2, 1, 3], modified_labels[1])

    def test_compute_metrics_for_qa_task(self):
        # given
        preds = np.random.random((2, 3, 4))
        labels = np.random.randint(2, 3)
        dataset = MagicMock()

        compute_legacy_metrics_for_sl_mock = MagicMock()

        # then
        with patch("ibformers.data.metrics.compute_legacy_metrics_for_sl", compute_legacy_metrics_for_sl_mock):
            metrics.compute_metrics_for_qa_task((preds, labels), dataset)

        # verify
        expected_labels = ["class_0", "class_1", "class_2", "class_3"]
        compute_legacy_metrics_for_sl_mock.assert_called_once_with((preds, labels), dataset, expected_labels)
