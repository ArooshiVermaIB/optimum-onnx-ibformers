from dataclasses import dataclass
from typing import Dict

import torch

from ibformers.data.collators.augmenters.base import BaseAugmenter


@dataclass
class BboxMaskingAugmenter(BaseAugmenter):
    """
    Augmenter for masked bbox prediction training task.
    """

    bbox_masking_probability: float = 0.15
    bbox_masking_whole_word_mask: float = False
    bbox_masking_value: int = None
    bbox_masking_replace_percentage: float = 1.0

    def __post_init__(self):
        if self.bbox_masking_value is None:
            self.bbox_masking_value = getattr(self.model.config, "max_2d_position_embeddings", 1024) - 1

    def augment(self, batch: Dict[str, torch.Tensor]):
        bbox = batch["bbox"]
        bbox_labels = batch["bbox"].clone()

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in batch["input_ids"].tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        base_shape = bbox_labels.shape[:-1]
        probability_matrix = torch.full(base_shape, self.bbox_masking_probability)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        bbox_labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # most of the time, we replace masked bbox with mask value
        # in the paper this is all the time (bbox_masking_replace_percentage = 1)
        indices_replaced = (
            torch.bernoulli(torch.full(base_shape, self.bbox_masking_replace_percentage)).bool() & masked_indices
        )
        bbox[indices_replaced] = self.bbox_masking_value

        # in other cases, similarly to MLM, we leave the bboxes as they are

        batch["bbox"] = bbox
        batch["bbox_labels"] = bbox_labels
        return batch
