from dataclasses import dataclass
from typing import ClassVar, List, Any, Optional

import numpy as np

from ibformers.data.collators.collators.base import BaseCollator


@dataclass
class DefaultValueCollator(BaseCollator):
    """
    Base data collator for fields with single value imputation for padded indices.

    This collator extends each feature from `_supported_fields` up to the desired length, with extra
    indices filled with `_default_value`.
    To create a specific DefaultValueCollator, subclass this class and set the class attributes
    to desired values. See ibformers.data.collators.collators.constant_value.BboxCollator for reference.

    Class Attributes:
        _supported_fields: The names of the fields supported by the collator.
        _default_value: The default value for extended indices.
    """

    _supported_fields: ClassVar[List[str]] = []
    _default_value: ClassVar[Any]

    @property
    def supported_fields(self) -> List[str]:
        return self._supported_fields

    @property
    def default_value(self) -> Any:
        return self._default_value

    def _collate_features(self, features, target_length: Optional[int] = None):
        feature_keys = self._get_feature_keys(features)
        assert any(
            f in feature_keys for f in self.supported_fields
        ), f"Neither of {self.supported_fields} columns was found in the inputs"
        present_supported_names = [key for key in feature_keys if key in self.supported_fields]
        batch = {}
        for feature_name in present_supported_names:
            feature_batch = [feature[feature_name] for feature in features]

            if target_length is None:
                target_length = max(len(feature) for feature in feature_batch)

            assert len(feature_batch) > 0, "Empty batch cannot be collated"

            if not isinstance(feature_batch[0], np.ndarray):
                feature_batch = [np.array(feat) for feat in feature_batch]

            feat_lst = []
            for feature in feature_batch:
                pad_width = target_length - len(feature)
                if pad_width > 0:
                    new_shape = [pad_width] + list(feature.shape[1:])
                    feat_lst.append(
                        np.concatenate((feature, np.full_like(feature, fill_value=self.default_value, shape=new_shape)))
                    )
                else:
                    feat_lst.append(feature)
            batch[feature_name] = feat_lst

        return batch


@dataclass
class BboxCollator(DefaultValueCollator):
    _supported_fields: ClassVar[List[str]] = ["bbox", "bboxes"]
    _default_value: ClassVar[Any] = 0


@dataclass
class TokenClassLabelCollator(DefaultValueCollator):
    _supported_fields: ClassVar[List[str]] = [
        "label",
        "labels",
        "token_order_ids",
        "chattered_row_ids",
        "chattered_col_ids",
        "token_row_ids",
        "token_col_ids",
    ]
    _default_value: ClassVar[Any] = -100


@dataclass
class StackedTableLabelCollator(DefaultValueCollator):
    _supported_fields: ClassVar[List[str]] = ["stacked_table_labels"]
    _default_value: ClassVar[Any] = -100


@dataclass
class MqaIdsCollator(DefaultValueCollator):
    _supported_fields: ClassVar[List[str]] = ["mqa_ids"]
    _default_value: ClassVar[Any] = 1


@dataclass
class AnswerTokenLabCollator(DefaultValueCollator):
    _supported_fields: ClassVar[List[str]] = ["answer_token_label_ids"]
    _default_value: ClassVar[Any] = -100


@dataclass
class QuestionPosCollator(DefaultValueCollator):
    _supported_fields: ClassVar[List[str]] = ["question_positions"]
    _default_value: ClassVar[Any] = 0


@dataclass
class QAPosCollator(BaseCollator):
    @property
    def supported_fields(self) -> List[str]:
        return ["start_positions", "end_positions"]

    def _collate_features(self, features, target_length: Optional[int] = None):
        feature_keys = self._get_feature_keys(features)
        assert any(
            f in feature_keys for f in self.supported_fields
        ), f"Neither of {self.supported_fields} columns was found in the inputs"
        present_supported_names = [key for key in feature_keys if key in self.supported_fields]
        batch = {}
        for feature_name in present_supported_names:
            feature_batch = [feature[feature_name] for feature in features]
            batch[feature_name] = feature_batch
        return batch
