from collections import defaultdict
from typing import Sequence, Type, List, Dict, Any, Optional

import torch
import numpy as np

from ibformers.data.collators.collators import BboxCollator
from ibformers.data.collators.collators.base import BaseCollator, CollatorABC
from ibformers.data.collators.collators.constant_value import PageMarginCollator, BboxShiftVectorCollator
from ibformers.data.collators.collators.image import DetrSubImageImageExtractor
from ibformers.data.collators.collators.table_structure import TableStructureBboxCollator, StructureObjectCollator
from ibformers.data.collators.collators.universal import safe_dataclass_init
from ibformers.third_party.detr.util.misc import NestedTensor


class TableDetrCollator(CollatorABC):
    collator_classes: List[Type[BaseCollator]] = [
        DetrSubImageImageExtractor,
        BboxCollator,
        StructureObjectCollator,
        TableStructureBboxCollator,
        PageMarginCollator,
        BboxShiftVectorCollator,
    ]

    ALREADY_TENSORIZED = ["detection_boxes", "detection_labels", "table_label_ids",
                          "structure_boxes", "structure_labels"]

    def __init__(
        self,
        **kwargs,
    ):
        """
        Args:
            padding: Options for padding
            max_length: Maximum length of the collated batch
            pad_to_multiple_of: If set, the collated batch will have dimensions that are a multiple
            of provided value. Useful for fp16 training.
            **kwargs: Extra arguments passed to additional collators.
        """
        self.collators: List[BaseCollator] = [
            safe_dataclass_init(
                collator,
                **kwargs,
            )
            for collator in self.collator_classes
        ]

        field_to_collator: Dict[str, List[BaseCollator]] = defaultdict(list)

        for collator in self.collators:
            for field in collator.supported_fields:
                field_to_collator[field].append(collator)
        self.field_to_collator = dict(field_to_collator)

    @property
    def supported_fields(self) -> List[str]:
        return list(self.field_to_collator.keys())

    def _get_collator_for_field(self, field_name: str) -> BaseCollator:
        collators = self.field_to_collator[field_name]
        if len(collators) > 1:
            raise RuntimeError(f"There are more than one collators for given filed.")
        try:
            return self.field_to_collator[field_name][0]
        except KeyError:
            raise KeyError(
                f"Could not find a collator for field {field_name}. "
                f"Make sure that the field name is correct, and define a subclass of "
                f"BaseCollator if it is not yet covered."
            )

    def _collate_features(self, features, target_length: Optional[int] = None):
        """
        Collate features into a batch.

        Calls BaseCollator first, and then extra collators for each remaining feature.

        Args:
            features: List of features

        Returns:
            Batch of collated tensors.

        Raises:
            KeyError: When a field name is present in features, but has no defined collator.
            RuntimeError: When a field name has multiple collators available.

        """
        to_collate = set(features[0].keys())
        first_collator = self.collators[0]
        collated = first_collator(features)
        to_collate -= set(first_collator.supported_fields)
        while len(to_collate) != 0:
            field = to_collate.pop()
            extra_collator = self._get_collator_for_field(field)
            extra_collated = extra_collator(features)
            to_collate -= set(extra_collator.supported_fields)
            collated = dict(**collated, **extra_collated)

        # TODO: custom types for fields
        batch = {k: self._maybe_tensorize(k, v) for k, v in collated.items()}
        return batch

    @classmethod
    def _maybe_tensorize(cls, key, value):
        if isinstance(value, (torch.Tensor, NestedTensor)):
            return value
        if key in cls.ALREADY_TENSORIZED:
            return value
        return torch.tensor(np.array(value))
