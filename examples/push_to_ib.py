from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer

from ibformers.trainer.hf_token import HF_TOKEN
from transformers.file_utils import PushToHubMixin
import fire
from ibformers.data.pipelines.pipeline import PIPELINES


def push_to_ib(model_path, pipeline_name, model_name):

    # load pipeline
    pipeline = PIPELINES[pipeline_name]
    model_class = pipeline['model_class']

    # load model
    model: PreTrainedModel = model_class.from_pretrained(model_path)
    if not hasattr(model.config, "pipeline_name"):
        model.config.pipeline_name = pipeline_name
    url = model.push_to_hub(
        model_name,
        commit_message='add model',
        organization='instabase',
        private=True,
        use_auth_token=HF_TOKEN,
    )

    # load tokenizer
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    url_tokenizer = tokenizer.push_to_hub(
        model_name,
        commit_message='add tokenizer',
        organization='instabase',
        private=True,
        use_auth_token=HF_TOKEN,
    )

    print(f"model uploaded. Model url: {url}. Tokenizer url should be the same: {url_tokenizer}")


if __name__ == '__main__':
    fire.Fire(push_to_ib)
