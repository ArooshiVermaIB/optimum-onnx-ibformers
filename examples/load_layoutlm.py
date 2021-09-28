from ibformers.models.layoutv2_noimg import LayoutLMv2NoImgModel
from transformers import LayoutLMv2Tokenizer
import torch
from pathlib import Path


tokenizer = LayoutLMv2Tokenizer.from_pretrained('microsoft/layoutlmv2-base-uncased', additional_special_tokens=[f'<extra_{i}>' for i in range(100)])
model = LayoutLMv2NoImgModel.from_pretrained('microsoft/layoutlmv2-base-uncased')


new_embeddings = model.resize_token_embeddings(len(tokenizer))

mask_token_id = tokenizer.mask_token_id
position_emb = model.embeddings.position_embeddings.weight.data

# use mask token as an initial weight for added tokens, after all it will be a bit similar usage

rand = torch.rand_like(new_embeddings.weight.data[-100:]) * 0.01
new_embeddings.weight.data[-100:] = new_embeddings.weight.data[mask_token_id:mask_token_id+1] + rand
model.embeddings.word_embeddings = new_embeddings

save_path = Path('/Users/rafalpowalski/python/models/layout-base-mqa')
save_path.mkdir(exist_ok=True, parents=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)











a = 1

