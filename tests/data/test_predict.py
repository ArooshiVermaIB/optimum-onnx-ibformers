import unittest

import numpy as np
from ibformers.data import predict


class TestPredict(unittest.TestCase):
    def test_doc_chunk_iter(self):
        # given
        doc_ids = ["1", "1", "1", "2", "2"]

        # then
        doc_ids = list(predict.doc_chunk_iter(doc_ids))

        # verify
        expected = [("1", 0, 3), ("2", 3, 5)]
        self.assertListEqual(doc_ids, expected)

    def test_doc_chunk_iter_single_doc_id(self):
        # given
        doc_ids = ["1"] * 10

        # then
        doc_ids = list(predict.doc_chunk_iter(doc_ids))

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
        chunks = predict.join_chunks(chunks, chunk_ranges)

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
        chunks = predict.join_chunks(chunks, chunk_ranges)

        # verify
        expected = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5])[:, None].repeat(2, axis=1)
        np.testing.assert_almost_equal(chunks, expected)

    def test_create_entities(self):
        # given
        predicted_entity_words = {
            "e1": [
                {"raw_word": "foo1", "conf": 0.5},
                {"raw_word": "bar1", "conf": 0.7},
            ],
            "e2": [
                {"raw_word": "bar2", "conf": 0.7},
            ],
        }
        gold_entity_words = {
            "e1": [
                {"raw_word": "foo1"},
                {"raw_word": "bar1"},
            ],
            "e2": [
                {"raw_word": "foo2"},
                {"raw_word": "bar2"},
            ],
        }
        label_list = ["0", "e1", "e2"]  # the first key is required to be an empty label

        # then
        entities = predict.create_entities(predicted_entity_words, gold_entity_words, label_list)

        # verify
        self.assertEquals(len(entities), 2)
        expected_e1 = {
            "words": predicted_entity_words["e1"],
            "text": "foo1 bar1",
            "avg_confidence": 0.6,
            "gold_text": "foo1 bar1",
            "gold_words": gold_entity_words["e1"],
            "is_match": True,
        }
        expected_e2 = {
            "words": predicted_entity_words["e2"],
            "text": "bar2",
            "avg_confidence": 0.7,
            "gold_text": "foo2 bar2",
            "gold_words": gold_entity_words["e2"],
            "is_match": False,
        }
        self.assertDictEqual(entities["e1"], expected_e1)
        self.assertDictEqual(entities["e2"], expected_e2)
