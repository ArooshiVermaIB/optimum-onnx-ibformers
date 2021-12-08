import string
import unittest
from typing import Any
from unittest.mock import MagicMock

from hypothesis import given, settings, reproduce_failure, Phase
from hypothesis import strategies as st
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.pipelines.base import Dataset

from ibformers.data.pipelines.pipeline import PIPELINES, prepare_dataset
from ibformers.trainer.hf_token import HF_TOKEN
from tests.resources.hypothesis_strategies import example, create_dataset_from_examples


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pipeline_model_id_pairs = [
            ("plain_sl", "bert-base-cased"),
            ("plain_mlm", "bert-base-cased"),
            ("layoutlm_sl", "microsoft/layoutlm-base-uncased"),
            ("layoutlmv2_sl", "microsoft/layoutlmv2-base-uncased"),
            ("laymqav1", "instabase/laymqav1-base"),
            ("layoutlm_mlm", "microsoft/layoutlm-base-uncased"),
        ]
        cls.pipeline_tokenizer_pairs = [
            (pipeline_name, AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN))
            for pipeline_name, model_name in pipeline_model_id_pairs
        ]

    def _test_pipelines_on_dataset(self, dataset: Dataset, padding: Any, max_length: int, chunk_overlap: int):
        for pipeline_name, tokenizer in self.pipeline_tokenizer_pairs:
            # given
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
            data_collator = pipeline["collate"](
                tokenizer,
                model=model,
            )
            supported_fields = set(data_collator.collator.supported_fields)

            # then
            try:
                prepared_dataset = prepare_dataset(
                    dataset, pipeline, fn_kwargs=fn_kwargs, keep_in_memory=True, num_proc=1
                )
                ignored_columns = list(set(prepared_dataset.column_names) - supported_fields)
                prepared_dataset = prepared_dataset.remove_columns(ignored_columns)
                dataloader = DataLoader(
                    prepared_dataset,
                    batch_size=2,
                    collate_fn=data_collator,
                    num_workers=0,
                )
                for _ in dataloader:
                    # just check if the dataloader iterates correctly for now
                    pass
                    # TODO: add some property tests of dataloader batches (equal sequence length etc)
            except AssertionError:
                pass
                # Allow failing on assertions, as they indicate expected data problems.
                # TODO: rework the way we throw errors on invalid data and other expected issues.
                #  Perhaps we should use some predefined exceptions that are handled specifically on the product side

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
    @settings(
        max_examples=25,
        phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target],
        deadline=None,
        print_blob=True,
    )
    def test_pipeline(self, example_list):
        # given
        dataset = create_dataset_from_examples(example_list)

        # then
        self._test_pipelines_on_dataset(dataset, False, 32, 4)


if __name__ == "__main__":
    unittest.main()
