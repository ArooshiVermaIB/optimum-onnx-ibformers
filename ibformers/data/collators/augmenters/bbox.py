from dataclasses import dataclass
import random

from typing import Dict

import torch

from ibformers.data.collators.augmenters.base import BaseAugmenter


@dataclass
class BboxAugmenter(BaseAugmenter):
    bbox_augmenter_max_offset: int = 20
    bbox_augmenter_max_scale: float = 0.05

    def augment(self, batch: Dict[str, torch.Tensor]):
        offset = self.bbox_augmenter_max_offset
        scale = self.bbox_augmenter_max_scale
        if self.model.training:
            # do bbox augmentation
            bbox = batch["bbox"]
            non_zeros_idx = (bbox[:, :, 0] != 0).to(bbox.dtype)

            x_offset = y_offset = random.randrange(-offset, offset)
            x_scale = y_scale = random.uniform(1 - scale, 1 + scale)

            bbox[:, :, [0, 2]] = (
                (bbox[:, :, [0, 2]] * x_scale + x_offset).clamp(1, 999).to(dtype=bbox.dtype)
            )
            bbox[:, :, [1, 3]] = (
                (bbox[:, :, [1, 3]] * y_scale + y_offset).clamp(1, 999).to(dtype=bbox.dtype)
            )
            bbox = bbox * non_zeros_idx[:, :, None]
            batch["bbox"] = bbox
        return batch
