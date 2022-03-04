import unittest
from unittest.mock import MagicMock

import numpy as np
from ibformers.data import predict


class TestPredict(unittest.TestCase):
    def _dataset_mock(self):
        dataset_dict = {}
        dataset_dict["id"] = ["id", "id", "id3"]
        dataset_dict["chunk_ranges"] = [[0, 4], [2, 6], [4, 8]]
        dataset_dict["content_tokens_mask"] = [[True, True, True, True]] * 3
        dataset_dict["word_starts"] = [[True, True, True, True]] * 3
        dataset_dict["words"] = [{}] * 3

        def getitem(val):
            if isinstance(val, int):
                return {k: v[val] for k, v in dataset_dict.items()}
            elif isinstance(val, slice):
                return {k: v[val] for k, v in dataset_dict.items()}
            else:
                return dataset_dict[val]

        dataset_mock = MagicMock()
        dataset_mock.__getitem__.side_effect = getitem
        return dataset_mock

    def test_extract_dechunked_components(self):
        # given
        doc_id = "id"
        chunk_from_idx = 0
        chunk_to_idx = 2
        preds = np.random.random((3, 4, 2))
        labels = np.array([[0, 0, 0, 1], [0, 1, 1, 1], [1, 1, 0, 0]])
        predictions = (preds, labels)
        dataset = self._dataset_mock()

        # then
        dechunked_doc = predict.extract_dechunked_components(doc_id, chunk_from_idx, chunk_to_idx, predictions, dataset)

        # verify
        self.assertEqual(dechunked_doc["id"], doc_id)
        np.testing.assert_array_equal(dechunked_doc["gold_labels"], [0, 0, 0, 1, 1, 1])
        self.assertEqual(dechunked_doc["raw_predictions"].shape, (6, 2))
        np.testing.assert_array_equal(dechunked_doc["raw_predictions"][:2], preds[0][:2])
        np.testing.assert_array_equal(dechunked_doc["raw_predictions"][2:4], (preds[0][2:] + preds[1][:2]) / 2)
        np.testing.assert_array_equal(dechunked_doc["raw_predictions"][4:], preds[1][2:])

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

    def test_calculate_predictions(self):
        # given
        raw_predicitons = np.array(
            [
                [-100.0, -100.0, 100.0],
                [-2.0, 2.0, -2.0],
                [1.0, -1.0, -1.0],
            ]
        )

        # then
        predictions = predict.calculate_predictions(raw_predicitons)

        # verify
        np.testing.assert_array_equal(predictions["predicted_classes"], [2, 1, 0])
        np.testing.assert_array_almost_equal(predictions["prediction_confidences"], [1.0, 0.964, 0.787], decimal=3)

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
        label_list = ["O", "e1", "e2"]  # the first key is required to be an empty label

        # then
        entities = predict.create_entities(predicted_entity_words, gold_entity_words, label_list)

        # verify
        self.assertEqual(len(entities), 2)
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

    def _get_entity_words_inputs(self):
        predicted_classes = np.array([0, 0, 1, 1, 2, 2, 0, 0])
        doc = {
            "words": [f"w{i}" for i in range(8)],
            "prediction_confidences": [1.0, 1.0, 0.9, 0.9, 0.8, 0.8, 1.0, 1.0],
            "word_original_bboxes": [[0, 10, 5, 20]] * 8,
            "word_page_nums": [0, 0, 0, 1, 1, 2, 2, 2],
        }
        label_list = ["O", "E1", "E2"]
        return predicted_classes, doc, label_list

    def test_extract_entity_words(self):
        # given
        predicted_classes, doc, label_list = self._get_entity_words_inputs()

        # then
        entity_words = predict.extract_entity_words(predicted_classes, doc, label_list, False)
        entity_words_gold = predict.extract_entity_words(predicted_classes, doc, label_list, True)

        # verify
        expected = {
            "E1": [
                dict(
                    raw_word="w2",
                    start_x=0,
                    start_y=10,
                    end_x=5,
                    end_y=20,
                    line_height=10,
                    word_width=5,
                    page=0,
                    conf=0.9,
                    idx=2,
                ),
                dict(
                    raw_word="w3",
                    start_x=0,
                    start_y=10,
                    end_x=5,
                    end_y=20,
                    line_height=10,
                    word_width=5,
                    page=1,
                    conf=0.9,
                    idx=3,
                ),
            ],
            "E2": [
                dict(
                    raw_word="w4",
                    start_x=0,
                    start_y=10,
                    end_x=5,
                    end_y=20,
                    line_height=10,
                    word_width=5,
                    page=1,
                    conf=0.8,
                    idx=4,
                ),
                dict(
                    raw_word="w5",
                    start_x=0,
                    start_y=10,
                    end_x=5,
                    end_y=20,
                    line_height=10,
                    word_width=5,
                    page=2,
                    conf=0.8,
                    idx=5,
                ),
            ],
        }
        self.assertDictEqual(expected, entity_words)

    def test_extract_entity_words_is_gold(self):
        # given
        predicted_classes, doc, label_list = self._get_entity_words_inputs()

        # then
        entity_words = predict.extract_entity_words(predicted_classes, doc, label_list, True)

        # verify
        for entity_word_list in entity_words.values():
            for word in entity_word_list:
                self.assertEquals(word["conf"], 0)

    def test_extract_entity_words_no_bbox(self):
        # given
        predicted_classes, doc, label_list = self._get_entity_words_inputs()
        doc.pop("word_original_bboxes")

        # then
        entity_words = predict.extract_entity_words(predicted_classes, doc, label_list, False)

        # verify
        for entity_word_list in entity_words.values():
            for word in entity_word_list:
                self.assertEquals(word["start_x"], 0)
                self.assertEquals(word["start_y"], 0)
                self.assertEquals(word["end_x"], 0)
                self.assertEquals(word["end_y"], 0)

    def test_extract_entity_words_no_word_page_nums(self):
        # given
        predicted_classes, doc, label_list = self._get_entity_words_inputs()
        doc.pop("word_page_nums")

        # then
        entity_words = predict.extract_entity_words(predicted_classes, doc, label_list, False)

        # verify
        for entity_word_list in entity_words.values():
            for word in entity_word_list:
                self.assertEquals(word["page"], 0)
