from pathlib import Path

import torch
from transformers import LayoutLMTokenizer, AutoConfig, RobertaForMaskedLM, LayoutLMModel

from ibformers.models.layv1mqa import Layv1MQAForSentinelClassification


# load the first model first to modify it's embedding matrix
basemodel = LayoutLMModel.from_pretrained('microsoft/layoutlm-base-uncased')
tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased',
                                                additional_special_tokens=[f'<extra_{i}>' for i in range(100)])

new_embeddings = basemodel.resize_token_embeddings(len(tokenizer))

assert len(new_embeddings.weight) == len(basemodel.embeddings.word_embeddings.weight)
mask_token_id = tokenizer.mask_token_id
# use mask token as an initial weight average for added tokens, after all it will be a bit similar usage
rand = torch.rand_like(new_embeddings.weight.data[-100:]) * 0.01
new_embeddings.weight.data[-100:] = new_embeddings.weight.data[mask_token_id:mask_token_id + 1] + rand
basemodel.embeddings.word_embeddings = new_embeddings

basemodel.config.start_extra_id = len(tokenizer) - 100
basemodel.config.end_extra_id = len(tokenizer)

save_path = Path('/home/ib/models/layoutv1-base-mqa-base')
save_path.mkdir(exist_ok=True, parents=True)
basemodel.save_pretrained(save_path)

# load the model with classification weight from base model, so the weights could be tied with the correct emb matrix
class_model = Layv1MQAForSentinelClassification.from_pretrained('/home/ib/models/layoutv1-base-mqa-base')

save_path = Path('/home/ib/models/layoutv1-base-mqa')
save_path.mkdir(exist_ok=True, parents=True)
class_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
