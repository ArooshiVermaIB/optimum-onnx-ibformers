from typing import Any

from datasets import load_dataset, ClassLabel, Sequence
from transformers import AutoTokenizer

from ibformers.data.tokenize import tokenize


# TODO: find out if we can get rid of InstabaseSDK for opening ibdocs
from ibformers.data.transform import norm_bboxes_for_layoutlm


class InstabaseSDK:
    def __init__(self, file_client: Any, username: str):
        # these will be ignored
        self.file_client = file_client
        self.username = username

    def ibopen(self, path: str, mode: str = 'r') -> Any:
        return open(path, mode)

    def read_file(self, file_path: str) -> str:
        with open(file_path) as f:
            return f.read()

    def write_file(self, file_path: str, content: str):
        with open(file_path, 'w') as f:
            f.write(content)


sdk = InstabaseSDK(None, "rpowalski")
dataset = load_dataset(path='/Users/rafalpowalski/python/ibformers/ibformers/datasets/ibds', name='ibds',
                       data_files=['/Users/rafalpowalski/python/annotation/uber/UberEats.ibannotator'],
                       ibsdk=sdk)

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
tokenized_dataset = dataset.map(tokenize, batched=True, batch_size=32, fn_kwargs={"tokenizer": tokenizer})

blah = tokenized_dataset.map(norm_bboxes_for_layoutlm, batched=True, batch_size=32, fn_kwargs={"tokenizer": tokenizer})


blah2 = blah.rename_column("token_label_ids", "labels")




# TODO: prepare bboxes for layoutlm - normalize it to 1-1000 range


a = 1
