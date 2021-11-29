import unittest

import numpy as np

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

    def test_iou_score_empty_lists(self):
        # given
        y_true = {"e1": []}
        y_pred = {"e1": []}
        all_tags = ["e1"]

        # then
        iou_score = metrics.iou_score(y_true, y_pred, all_tags)

        # verify
        expected = {"e1": 0.0}
        self.assertDictEqual(iou_score, expected)
