from datasets import load_dataset, ClassLabel, Sequence
from transformers import AutoTokenizer

from ibformers.data.tokenize import tokenize
from ibformers.data.transform import fuzzy_tag_in_document, add_token_labels_qa

dataset = load_dataset(
    path="/Users/rafalpowalski/python/ibformers/ibformers/datasets/docvqa", name="docvqa", split="validation"
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
extractive_qa_dataset = dataset.map(fuzzy_tag_in_document, batched=True, batch_size=32)
tokenized_dataset = extractive_qa_dataset.map(tokenize, batched=True, batch_size=32, fn_kwargs={"tokenizer": tokenizer})
new_features = tokenized_dataset.features.copy()
new_features["token_label_ids"] = Sequence(ClassLabel(names=["O", "Answer"]))
final_ds = tokenized_dataset.map(add_token_labels_qa, batched=True, batch_size=32, features=new_features)

[fuzzy_tag_in_document, tokenize]
a = 1
