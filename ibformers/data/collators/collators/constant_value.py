from dataclasses import dataclass
from typing import ClassVar, List, Any, Optional

from ibformers.data.collators.collators.base import BaseCollator


@dataclass
class DefaultValueCollator(BaseCollator):
    """
    Base data collator for fields with single value imputation for padded indices.

    This collator extends each feature from `_padded_fields` up to the desired length, with extra
    indices filled with `_default_value`.
    To create a specific DefaultValueCollator, subclass this class and set the class attributes
    to desired values. See ibformers.data.collators.collators.constant_value.BboxCollator for reference.

    Class Attributes:
        _padded_fields: The names of the fields supported by the collator.
        _default_value: The default value for extended indices.
    """
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
