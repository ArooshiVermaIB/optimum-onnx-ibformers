import string
import unittest
from unittest.mock import MagicMock

from hypothesis import given, settings, Phase, reproduce_failure
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.pipelines.base import Dataset
from typing import Any

from ibformers.data.pipelines.pipeline import PIPELINES, prepare_dataset
from ibformers.trainer.hf_token import HF_TOKEN
from tests.resources.hypothesis_strategies import example, create_dataset_from_examples

from hypothesis import strategies as st


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline_tokenizer_pairs = [
            ("plain_sl", AutoTokenizer.from_pretrained("bert-base-cased")),
            ("plain_mlm", AutoTokenizer.from_pretrained("bert-base-cased")),
            ("layoutlm_sl", AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")),
            ("layoutlmv2_sl", AutoTokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")),
            ("laymqav1", AutoTokenizer.from_pretrained("instabase/laymqav1-base", use_auth_token=HF_TOKEN)),
            ("layoutlm_mlm", AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")),
        ]

    def _test_pipelines_on_dataset(self, dataset: Dataset, padding: Any, max_length: int, chunk_overlap: int):
        for pipeline_name, tokenizer in self.pipeline_tokenizer_pairs:
            pipeline = PIPELINES[pipeline_name]
            fn_kwargs = {
                "tokenizer": tokenizer,
                "padding": padding,
                "max_length": max_length,
                "chunk_overlap": chunk_overlap,
                "chain_functions": True,
            }
            model = MagicMock()
            model.training = True
            # TODO: finish with dataloader pass (remove ignored columns)
            # data_collator = pipeline["collate"](
            #     tokenizer,
            #     model=model,
            # )
            try:
                prepared_dataset = prepare_dataset(
                    dataset, pipeline, fn_kwargs=fn_kwargs, keep_in_memory=True, num_proc=1
                )
                pass
                # dataloader = DataLoader(
                #     prepared_dataset,
                #     batch_size=2,
                #     collate_fn=data_collator,
                #     num_workers=0,
                # )
                # for batch in dataloader:
                #     pass
            except AssertionError:
                pass  # Allow failing on assertions, as they indicate expected data problems.

    @given(
        st.lists(
            example(
                min_example_len=1,
                max_example_len=50,
                num_fields=4,
                allow_invalid_bboxes=False,
                allowed_text_characters=string.ascii_letters + string.digits + string.whitespace,
            ),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=25, deadline=None, print_blob=True)
    def test_pipeline(self, example_list):
        # given
        dataset = create_dataset_from_examples(example_list)

        # then
        self._test_pipelines_on_dataset(dataset, False, 32, 4)

    @given(
        st.lists(
            example(
                min_example_len=1,
                max_example_len=50,
                num_fields=4,
                allow_invalid_bboxes=True,
                allowed_text_characters=string.ascii_letters + string.digits,
            ),
            min_size=3,
            max_size=3,
        )
    )
    @settings(max_examples=10, deadline=None, print_blob=True)
    def test_pipeline_incorrect_bboxes(self, example_list):
        # given
        dataset = create_dataset_from_examples(example_list)

        # then
        self._test_pipelines_on_dataset(dataset, False, 32, 4)


if __name__ == "__main__":
    unittest.main()
