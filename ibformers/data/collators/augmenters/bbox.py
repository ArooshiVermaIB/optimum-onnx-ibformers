from dataclasses import dataclass
import random

from typing import Dict

import torch

from ibformers.data.collators.augmenters.base import BaseAugmenter


@dataclass
class BboxAugmenter(BaseAugmenter):
    """
    BoundingBox augmenter.

    All bounding boxes on the page are randomly shifted and scaled by the same value/factor.
    The parameters are sampled independently for x- and y-axis.

    Attributes:
        bbox_augmenter_max_offset: maximum absolute offset in pixels for bounding box shift. Note that maximum value
        for bbox coordinates for layoutlm models is 1000
        bbox_augmenter_max_scale: param for scale factor range. The actual range is (1-max_scale, 1+max_scale).
    """

    bbox_augmenter_max_offset: int = 20
    bbox_augmenter_max_scale: float = 0.05

    def augment(self, batch: Dict[str, torch.Tensor]):
        offset = self.bbox_augmenter_max_offset
        scale = self.bbox_augmenter_max_scale
        if self.model.training:
            # do bbox augmentation
            bbox = batch["bbox"]
            non_zeros_idx = (bbox[:, :, 0] != 0).to(bbox.dtype)

            x_offset = random.randrange(-offset, offset)
            y_offset = random.randrange(-offset, offset)
            x_scale = random.uniform(1 - scale, 1 + scale)
            y_scale = random.uniform(1 - scale, 1 + scale)

            bbox[:, :, [0, 2]] = (bbox[:, :, [0, 2]] * x_scale + x_offset).clamp(1, 999).to(dtype=bbox.dtype)
            bbox[:, :, [1, 3]] = (bbox[:, :, [1, 3]] * y_scale + y_offset).clamp(1, 999).to(dtype=bbox.dtype)
            bbox = bbox * non_zeros_idx[:, :, None]
            batch["bbox"] = bbox
        return batch
