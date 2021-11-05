from dataclasses import dataclass, field
from typing import ClassVar, List, Any, Optional

from ibformers.data.collators.collators.base import BaseCollator


@dataclass
class DefaultValueCollator(BaseCollator):
    _padded_fields: ClassVar[List[str]] = []
    _default_value: ClassVar[Any]

    @property
    def padded_fields(self) -> List[str]:
        return self._padded_fields

    @property
    def default_value(self) -> Any:
        return self._default_value

    def _collate_features(self, features, target_length: Optional[int] = None):
        feature_keys = self._get_feature_keys(features)
        assert (
            any(f in feature_keys for f in self.padded_fields)
        ), f"Neither of {self.padded_fields} columns was found in the inputs"
        present_feature_names = [key for key in feature_keys]
        batch = {}
        for feature_name in present_feature_names:
            feature_batch = [feature[feature_name] for feature in features]

            if target_length is None:
                target_length = max(len(feature) for feature in feature_batch)

            batch[feature_name] = [
                feature + [self._default_value] * (target_length - len(feature))
                for feature in feature_batch
            ]
        return batch


@dataclass
class BboxCollator(DefaultValueCollator):
    _padded_fields: ClassVar[List[str]] = ['bbox', 'bboxes']
    _default_value: ClassVar[Any] = [0, 0, 0, 0]


@dataclass
class TokenClassLabelCollator(DefaultValueCollator):
    _padded_fields: ClassVar[List[str]] = ['label', 'labels']
    _default_value: ClassVar[Any] = -100
