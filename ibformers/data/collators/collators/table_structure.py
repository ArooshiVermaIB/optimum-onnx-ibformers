from dataclasses import dataclass
from typing import List, ClassVar, Optional

import torch

from ibformers.data.collators.collators.base import CollatorABC


@dataclass
class StructureObjectCollator(CollatorABC):
    _supported_fields: ClassVar[List[str]] = [
        "structure_labels",
        "detection_labels",
        "table_label_ids",
    ]

    @property
    def supported_fields(self) -> List[str]:
        return self._supported_fields

    def _collate_features(self, features, target_length: Optional[int] = None):
        feature_keys = list(features[0].keys())
        assert any(
            f in feature_keys for f in self.supported_fields
        ), f"Neither of {self.supported_fields} columns was found in the inputs"
        present_supported_names = [key for key in feature_keys if key in self.supported_fields]
        batch = {}
        for feature_name in present_supported_names:
            feature_batch = [torch.tensor(feature[feature_name], dtype=torch.long) for feature in features]
            batch[feature_name] = feature_batch
        return batch


@dataclass
class TableStructureBboxCollator(CollatorABC):
    _supported_fields: ClassVar[List[str]] = [
        "detection_boxes",
        "structure_boxes",
    ]

    @property
    def supported_fields(self) -> List[str]:
        return self._supported_fields

    def _collate_features(self, features, target_length: Optional[int] = None):
        feature_keys = list(features[0].keys())
        assert any(
            f in feature_keys for f in self.supported_fields
        ), f"Neither of {self.supported_fields} columns was found in the inputs"
        present_supported_names = [key for key in feature_keys if key in self.supported_fields]
        batch = {}
        for feature_name in present_supported_names:
            feature_batch = [
                torch.tensor(feature[feature_name], dtype=torch.float).reshape(len(feature[feature_name]), 4)
                for feature in features
            ]
            batch[feature_name] = feature_batch
        return batch
