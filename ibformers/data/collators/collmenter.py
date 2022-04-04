from typing import Union, Optional, List, Type, Dict, Any, Callable

from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from transformers.data.data_collator import InputDataClass
from transformers.file_utils import PaddingStrategy

from ibformers.data.collators.augmenters import BboxAugmenter, MLMAugmenter, BboxMaskingAugmenter
from ibformers.data.collators.augmenters.base import BaseAugmenter, AugmenterManager
from ibformers.data.collators.collators.universal import UniversalDataCollator


AUGMENTERS = {
    "bbox": BboxAugmenter,
    "mlm": MLMAugmenter,
    "bbox_masking": BboxMaskingAugmenter,
}


class CollatorWithAugmentation(Callable[[List[InputDataClass]], Dict[str, Any]]):
    """
    A joint class for feature collation and data augmentation.

    While it is not the most convenient to keep both those functionalities in single class,
    it fits nicely into transformer's Trainer.

    the instance is initialized in the training loop.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        augmenters_list: Optional[List[str]] = None,
        **augmenter_kwargs: Dict,
    ):
        """
        Args:
            tokenizer: Tokenizer for the model that the data is augmented for.
            model: Model for the training task
            padding: Options for padding
            max_length: Maximum length of the collated batch
            pad_to_multiple_of: If set, the collated batch will have dimensions that are a multiple
            of provided value. Useful for fp16 training.
            augmenter_kwargs: Additional arguments to pass to the augmenters.
        """
        self.augmenters_list = augmenters_list
        self.augmenters_to_use = self.get_augmenters(augmenters_list)

        self.collator = UniversalDataCollator(
            tokenizer=tokenizer, padding=padding, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of
        )
        self.augmenter = AugmenterManager(
            tokenizer=tokenizer, model=model, augmenters_to_use=self.augmenters_to_use, **augmenter_kwargs
        )

    def __call__(self, features) -> Dict[str, Any]:
        batch = self.collator(features)
        return self.augmenter.augment(batch)

    @staticmethod
    def get_augmenters(aug_list: Optional[List[str]]):
        if aug_list is None:
            return []
        augs = []
        for aug_name in aug_list:
            if aug_name not in AUGMENTERS:
                raise ValueError(f"{aug_name} augmentator is not supported")
            aug = AUGMENTERS[aug_name]
            augs.append(aug)
        return augs
