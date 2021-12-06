import unittest

from transformers import AutoTokenizer

from ibformers.data import tokenize


class TestTokenize(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def test_tokenize(self):
        # undecorate
        fn_to_test = tokenize.tokenize.__closure__[0].cell_contents

        # given
        batch = {"words": [["foo", "bar"]], "token_label_ids": [[0, 0]]}

        # then
        tokenized = fn_to_test(batch, self.tokenizer)

        # verify
        self.assertEqual(len(tokenized["input_ids"]), 1)
        self.assertEqual(len(tokenized["token_label_ids"]), 1)

        self.assertEqual(len(tokenized["input_ids"][0]), 2)
        self.assertEqual(len(tokenized["token_label_ids"][0]), 2)

    def test_tokenize_with_whitespace(self):
        # undecorate
        fn_to_test = tokenize.tokenize.__closure__[0].cell_contents

        # given
        batch = {"words": [["foo", "\xa0", "bar"]], "token_label_ids": [[0, 0, 0]]}

        # then
        tokenized = fn_to_test(batch, self.tokenizer)  # TODO: this shouldn't throw an error

        # verify
        self.assertEqual(len(tokenized["input_ids"]), 1)
        self.assertEqual(len(tokenized["token_label_ids"]), 1)

        self.assertEqual(len(tokenized["input_ids"][0]), 2)
        self.assertEqual(len(tokenized["token_label_ids"][0]), 2)
