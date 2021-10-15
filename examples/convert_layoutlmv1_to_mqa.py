from pathlib import Path

import torch
from transformers import LayoutLMTokenizer, AutoConfig, RobertaForMaskedLM, LayoutLMModel


# load the first model first to modify it's embedding matrix
from ibformers.models.layv1mqa import LayMQAModel, LayMQAForTokenClassification

config = AutoConfig.from_pretrained('microsoft/layoutlm-base-uncased')
size = 20
config.mqa_size = size
config.pad_mqa_id = 1

tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
tokenizer.mqa_size = size
tokenizer.pad_mqa_id = 1
class_model = LayMQAForTokenClassification.from_pretrained('microsoft/layoutlm-base-uncased', config=config)

save_path = Path('/home/ib/models/layoutv1-base-ttmqa')
save_path.mkdir(exist_ok=True, parents=True)
class_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
