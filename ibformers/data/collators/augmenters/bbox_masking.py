from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformers import DataCollatorForWholeWordMask

from ibformers.data.collators.augmenters.base import BaseAugmenter


@dataclass
class BboxMaskingAugmenter(BaseAugmenter):
    """
    Augmenter for masked bbox prediction training task.

    Attributes:
        bbox_masking_probability: the probability for the token to be masked
        bbox_masking_whole_word_mask: If True, whole un-tokenized tokens are masked.
        bbox_masking_value: Custom value for masked bbox coordinate
        bbox_masking_replace_percentage: The probability that masked token will be replaced by masking value.
         Allows similar flow to MLM, where part of the tokens selected for masking will be left unmasked.
    """

    bbox_masking_probability: float = 0.15
    bbox_masking_whole_word_mask: bool = False
    bbox_masking_value: int = None
    bbox_masking_replace_percentage: float = 1.0

    def __post_init__(self):
        if self.bbox_masking_value is None:
            self.bbox_masking_value = getattr(self.model.config, "max_2d_position_embeddings", 1024) - 1
        self.wwm_collator = DataCollatorForWholeWordMask(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=self.bbox_masking_probability
        )

    def augment(self, batch: Dict[str, torch.Tensor]):
        if self.bbox_masking_whole_word_mask:
            masked_indices = self._get_mask_wwm(batch)
        else:
            masked_indices = self._get_mask(batch)

        return self._mask_tokens(batch, masked_indices)

    def _get_mask(self, batch: Dict[str, torch.Tensor]):
        bbox_labels = batch["bbox"]
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in batch["input_ids"].tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix = torch.full(bbox_labels.shape[:-1], self.bbox_masking_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        return masked_indices

    def _get_mask_wwm(self, batch: Dict[str, torch.Tensor]):
        # workaround: let the MLM collator prepare masks, and then propagate them onto the bboxes

        # depending whether the MLM augmentation is before or after
        if "labels" in batch:
            input_col = torch.where(batch["labels"] == -100, batch["input_ids"], batch["labels"])
        else:
            input_col = batch["input_ids"]
        out_dict = self.wwm_collator.torch_call(input_col.tolist())

        return out_dict["input_ids"] == self.tokenizer.mask_token_id

    def _mask_tokens(self, batch: Dict[str, torch.Tensor], masked_indices: torch.Tensor):
        bbox = batch["bbox"]
        bbox_labels = batch["bbox"].clone()
        base_shape = bbox_labels.shape[:-1]
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
