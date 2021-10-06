from pathlib import Path

import torch
from transformers import LayoutLMv2Tokenizer, AutoConfig, RobertaForMaskedLM

from ibformers.models.laymqa import LayMQAModel, LayMQAForSentinelClassification


# load the first model first to modify it's embedding matrix
basemodel = LayMQAModel.from_pretrained('microsoft/layoutlmv2-base-uncased')
tokenizer = LayoutLMv2Tokenizer.from_pretrained('microsoft/layoutlmv2-base-uncased',
                                                additional_special_tokens=[f'<extra_{i}>' for i in range(100)])

mask_token_id = tokenizer.mask_token_id
# use mask token as an initial weight average for added tokens, after all it will be a bit similar usage
rand = torch.rand_like(new_embeddings.weight.data[-100:]) * 0.01
new_embeddings.weight.data[-100:] = new_embeddings.weight.data[mask_token_id:mask_token_id + 1] + rand
basemodel.embeddings.word_embeddings = new_embeddings

basemodel.config.start_extra_id = len(tokenizer) - 100
basemodel.config.end_extra_id = len(tokenizer)

save_path = Path('/Users/rafalpowalski/python/models/layout-base-mqa-base')
save_path.mkdir(exist_ok=True, parents=True)
basemodel.save_pretrained(save_path)

# load the model with classification weight from base model, so the weights could be tied with the correct emb matrix
class_model = LayMQAForSentinelClassification.from_pretrained('/Users/rafalpowalski/python/models/layout-base-mqa-base')

save_path = Path('/Users/rafalpowalski/python/models/layout-base-mqa')
save_path.mkdir(exist_ok=True, parents=True)
class_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
