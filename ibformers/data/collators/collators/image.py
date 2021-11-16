from dataclasses import dataclass
from typing import ClassVar, List, Optional

from ibformers.data.collators.collators.base import BaseCollator


@dataclass
class ImageCollator(BaseCollator):
    _supported_fields: ClassVar[List[str]] = ['image']

    @property
    def supported_fields(self) -> List[str]:
        return self._supported_fields

    def _collate_features(self, features, target_length: Optional[int] = None):
        feature_keys = self._get_feature_keys(features)
        assert any(
            f in feature_keys for f in self.supported_fields
        ), f"Neither of {self.supported_fields} columns was found in the inputs"
        present_supported_names = [key for key in feature_keys if key in self.supported_fields]
        batch = {}
        for feature_name in present_supported_names:
            assert all(
                feature[feature_name].shape[0] == 1 for feature in features
            ), "Collator supports only single pages"
            feature_batch = [feature[feature_name][0] for feature in features]
            batch[feature_name] = feature_batch
        return batch
