from dataclasses import dataclass
import random

from typing import Dict

import torch
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask

from ibformers.data.collators.augmenters.base import BaseAugmenter


@dataclass
class MLMAugmenter(BaseAugmenter):
    mlm_augmenter_probability: float = 0.15
    mlm_augmenter_whole_word_mask: bool = True

    def __post_init__(self):
        non_wwm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mlm_augmenter_probability
        )
        wwm_collator = DataCollatorForWholeWordMask(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mlm_augmenter_probability
        )
        self.collator = wwm_collator if self.mlm_augmenter_probability else non_wwm_collator

    def augment(self, batch: Dict[str, torch.Tensor]):
        out_dict = self.collator.torch_call(batch['input_ids'].tolist())

        batch['input_ids'] = out_dict['input_ids']
        batch['labels'] = out_dict['labels']
        return batch
