import unittest
from unittest.mock import patch

from transformers import AutoTokenizer

from ibformers.data import chunk
import numpy as np


class TestChunk(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def _get_test_example(self, get_images: bool, get_mqa_features: bool):
        # TODO: replace by hypothesis generators
        example = {
            "input_ids": list(range(1, 11)),
            "bboxes": [(i, i + 1, i + 2, i + 3) for i in range(10)],
            "token_label_ids": [0] * 5 + [1] * 5,
        }
        if get_images:
            example["token_page_nums"] = [0] * 5 + [1] * 5
            example["images_page_nums"] = [0, 1]
            example["images"] = np.random.randint(0, 255, (2, 3, 224, 244))
        if get_mqa_features:
            example["prefix_input_ids"] = [1, 2]
            example["prefix_mqa_ids"] = [0, 1]
        return example

    def test_get_chunk_ranges(self):
        # given
        input_len = 200
        chunk_size = 50
        overlap = 20

        # then
        ranges = chunk.get_chunk_ranges(input_len, chunk_size, overlap)

        # verify
        expected = [(0, 50), (30, 80), (60, 110), (90, 140), (120, 170), (150, 200)]
        self.assertListEqual(ranges, expected)

    def test_get_chunk_ranges_high_chunk_size(self):
        # given
        input_len = 200
        chunk_size = 250
        overlap = 20

        # then
        ranges = chunk.get_chunk_ranges(input_len, chunk_size, overlap)

        # verify
        expected = [(0, 200)]
        self.assertListEqual(ranges, expected)

    def test_get_chunk_ranges_no_overlap(self):
        # given
        input_len = 200
        chunk_size = 50
        overlap = 0

        # then
        ranges = chunk.get_chunk_ranges(input_len, chunk_size, overlap)

        # verify
        expected = [(0, 50), (50, 100), (100, 150), (150, 200)]
        self.assertListEqual(ranges, expected)

    def test_get_chunk_ranges_low_size(self):
        # given
        input_len = 10
        chunk_size = 3
        overlap = 0

        # then
        ranges = chunk.get_chunk_ranges(input_len, chunk_size, overlap)

        # verify
        expected = [(0, 3), (3, 6), (6, 9), (9, 10)]
        self.assertListEqual(ranges, expected)

    def test_get_single_page_chunk_ranges(self):
        # given
        input_len = 200
        chunk_size = 50
        overlap = 20
        page_nums = [0] * 30 + [1] * 170

        # then
        ranges = chunk.get_single_page_chunk_ranges(input_len, chunk_size, overlap, page_nums)

        # verify
        expected = [(0, 30), (30, 80), (60, 110), (90, 140), (120, 170), (150, 200)]
        self.assertListEqual(ranges, expected)

    def test_get_single_page_chunk_ranges_incorrect_page_nums_length(self):
        # given
        input_len = 200
        chunk_size = 50
        overlap = 20
        page_nums = [0] * 30 + [1] * 200

        # verify
        with self.assertRaises(AssertionError):
            ranges = chunk.get_single_page_chunk_ranges(input_len, chunk_size, overlap, page_nums)

    def test_fill_special_tokens(self):
        # given
        arr = [1, 2, 3, 4]
        content_mask = [False] + [True] * 4 + [False]
        fill_value = 0

        # then
        filled = chunk.fill_special_tokens(arr, content_mask, fill_value)

        # verify
        expected = [fill_value] + arr + [fill_value]
        self.assertListEqual(filled.tolist(), expected)

    def test_fill_special_tokens_multi_dims(self):
        # given
        arr = [[1, 2, 3, 4], [2, 3, 4, 5]]
        content_mask = [False] + [True] * 2 + [False]
        fill_value = 0

        # then
        filled = chunk.fill_special_tokens(arr, content_mask, fill_value)

        # verify
        expected = [[fill_value] * 4] + arr + [[fill_value] * 4]
        self.assertListEqual(filled.tolist(), expected)

    def test_split_by_ranges(self):
        # given
        seq = list(range(10))
        ranges = [(0, 5), (5, 10)]

        # then
        split_ = chunk._split_by_ranges(seq, ranges)

        # verify
        expected = [list(range(5)), list(range(5, 10))]
        self.assertListEqual(split_, expected)

    def test_split_by_ranges_wrong_length(self):
        # given
        seq = list(range(10))
        ranges = [(0, 5), (5, 9)]

        # verify
        with self.assertRaises(ValueError):
            chunk._split_by_ranges(seq, ranges)

    def test_get_chunks(self):
        # given
        example = self._get_test_example(False, False)
        tokenizer = self.tokenizer
        chunk_ranges = [(0, 5), (5, 10)]

        # then
        chunks = list(chunk.get_chunks(example, tokenizer, chunk_ranges))

        # verify
        self.assertEquals(len(chunks), 2)
        keys_to_check = [k for k in example.keys() if k in chunk.KEYS_TO_CHUNK]
        for chunk_ in chunks:
            for key in keys_to_check:
                self.assertEquals(len(chunk_[key]), len(chunk_[keys_to_check[0]]))
                # TODO: not all KEYS_TO_CHUNK are actualle the same length.
                #  Some are chunked, but are not filled with the extra tokens (e.g. token_page_nums)

    def test_get_chunks_mqa(self):
        # given
        example = self._get_test_example(False, True)
        tokenizer = self.tokenizer
        chunk_ranges = [(0, 5), (5, 10)]

        # then
        chunks = list(chunk.get_chunks(example, tokenizer, chunk_ranges))

        # verify
        self.assertEquals(len(chunks), 2)
        keys_to_check = [k for k in example.keys() if k in chunk.KEYS_TO_CHUNK]
        for chunk_ in chunks:
            for key in keys_to_check:
                self.assertEquals(len(chunk_[key]), len(chunk_[keys_to_check[0]]))

    def test_get_chunks_with_image(self):
        # given
        example = self._get_test_example(True, True)
        tokenizer = self.tokenizer
        chunk_ranges = [(0, 5), (5, 8), (8, 10)]

        # then
        chunks = list(chunk.get_chunks(example, tokenizer, chunk_ranges))

        # verify
        self.assertEquals(len(chunks), 3)

    def test_produce_chunks_all_chunks(self):
        # undecorate
        fn_to_test = chunk.produce_chunks.__closure__[0].cell_contents
        # given
        example = self._get_test_example(False, True)
        tokenizer = self.tokenizer
        max_length = 5
        chunk_overlap = 0
        chunking_strategy = "ALL_CHUNKS"

        # then
        chunks = list(fn_to_test(example, tokenizer, max_length, chunking_strategy, chunk_overlap))

        # verify
        for chunk_ in chunks:
            self.assertLessEqual(len(chunk_["input_ids"]), max_length)

    def test_produce_chunks_single_page_chunks(self):
        # undecorate
        fn_to_test = chunk.produce_chunks.__closure__[0].cell_contents
        # given
        example = self._get_test_example(True, True)
        tokenizer = self.tokenizer
        max_length = 5
        chunk_overlap = 0
        chunking_strategy = "SINGLE_PAGES"

        # then
        chunks = list(fn_to_test(example, tokenizer, max_length, chunking_strategy, chunk_overlap))

        # verify
        for chunk_ in chunks:
            self.assertLessEqual(len(chunk_["input_ids"]), max_length)

    def test_produce_chunks_wrong_strategy(self):
        # undecorate
        fn_to_test = chunk.produce_chunks.__closure__[0].cell_contents
        # given
        example = self._get_test_example(True, True)
        tokenizer = self.tokenizer
        max_length = 5
        chunk_overlap = 0
        chunking_strategy = "MISSING_STRATEGY"

        # verify
        with self.assertRaises(ValueError):
            list(fn_to_test(example, tokenizer, max_length, chunking_strategy, chunk_overlap))
