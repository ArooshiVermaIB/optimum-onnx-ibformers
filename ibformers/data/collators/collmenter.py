# TODO: better name
import random
from dataclasses import dataclass
from typing import Union, Optional, ClassVar, List, Type, Dict, Any, Callable

import torch
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
    DataCollatorForTokenClassification, DataCollator,
)
from transformers.data.data_collator import InputDataClass
from transformers.file_utils import PaddingStrategy

from ibformers.data.collators.augmenters.base import BaseAugmenter, Augmenter
from ibformers.data.collators.collators.universal import UniversalDataCollator


class CollatorWithAugmentation(Callable[[List[InputDataClass]], Dict[str, Any]]):
    augmenters_to_use: List[Type[BaseAugmenter]]

    def __init__(self, tokenizer: PreTrainedTokenizerBase,
                 model: PreTrainedModel,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 max_length: Optional[int] = None,
                 pad_to_multiple_of: Optional[int] = None,
                 augmenter_kwargs: Optional[Dict] = None):
        self.collator = UniversalDataCollator(
            tokenizer=tokenizer,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of
        )
        augmenter_kwargs = {} if augmenter_kwargs is None else augmenter_kwargs
        self.augmenter = Augmenter(
            tokenizer=tokenizer,
            model=model,
            augmenters_to_use=self.augmenters_to_use,
            **augmenter_kwargs
        )

    def __call__(self, features) -> Dict[str, Any]:
        batch = self.collator(features)
        return self.augmenter.augment(batch)


def get_collator_class(*augmenters_to_use: Type[BaseAugmenter]):
    return type("CustomCollatorAugmenter", (CollatorWithAugmentation, ),
                {'augmenters_to_use': list(augmenters_to_use)})
