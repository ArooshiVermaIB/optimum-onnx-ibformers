from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, List, ClassVar

from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


class CollatorABC(ABC):
    @abstractmethod
    def _collate_features(self, features, target_length: Optional[int] = None):
        pass

    @property
    @abstractmethod
    def supported_fields(self):
        pass

    def __call__(self, features, target_length: Optional[int] = None):
        all_features = self._collate_features(features, target_length)
        return {
            feature_name: feature_value
            for (feature_name, feature_value) in all_features.items()
            if feature_name in self.supported_fields
        }


@dataclass
class BaseCollator(CollatorABC):
    """
    Base data collator. Uses provided tokenizer for data collation.

    We assume that this collator should be called for each model.
    Each subclass of this class gets registered in `extra_collators` field.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    extra_collators: ClassVar[List["BaseCollator"]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.extra_collators.append(cls)

    def __post_init__(self):
        if self.tokenizer.padding_side != "right":
            raise ValueError("Only right padding is supported")

    @staticmethod
    def _get_feature_keys(features) -> List[str]:
        return list(features[0].keys())

    @property
    def supported_fields(self):
        return self.tokenizer.model_input_names

    def _collate_features(self, features, target_length: Optional[int] = None):
        return self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors=None,
        )
