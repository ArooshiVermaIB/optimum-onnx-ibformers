import unittest

import numpy as np

from ibformers.data import metrics


class TestMetrics(unittest.TestCase):
    def test_doc_chunk_iter(self):
        # given
        doc_ids = ["1", "1", "1", "2", "2"]

        # then
        doc_ids = list(metrics.doc_chunk_iter(doc_ids))

        # verify
        expected = [("1", 0, 3), ("2", 3, 5)]
        self.assertListEqual(doc_ids, expected)

    def test_doc_chunk_iter_single_doc_id(self):
        # given
        doc_ids = ["1"] * 10

        # then
        doc_ids = list(metrics.doc_chunk_iter(doc_ids))

        # verify
        expected = [
            ("1", 0, 10),
        ]
        self.assertListEqual(doc_ids, expected)

    def test_join_chunks_1d(self):
        # given
        chunks = np.array(
            [
                [0.1, 0.1, 0.1, 0.1],
                [0.3, 0.3, 0.3, 0.3],
                [0.5, 0.5, 0.5, 0.5],
            ],
            dtype=np.float,
        )
        chunk_ranges = [(0, 4), (2, 6), (5, 9)]

        # then
        chunks = metrics.join_chunks(chunks, chunk_ranges)

        # verify
        expected = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5])
        np.testing.assert_almost_equal(chunks, expected)

    def test_join_chunks_2d(self):
        # given
        chunks = np.array([[0.1, 0.1, 0.1, 0.1], [0.3, 0.3, 0.3, 0.3], [0.5, 0.5, 0.5, 0.5]], dtype=np.float)[
            :, :, None
        ].repeat(2, axis=2)
        chunk_ranges = [(0, 4), (2, 6), (5, 9)]

        # then
        chunks = metrics.join_chunks(chunks, chunk_ranges)

        # verify
        expected = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5])[:, None].repeat(2, axis=1)
        np.testing.assert_almost_equal(chunks, expected)

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
