from collections import defaultdict
from dataclasses import fields, dataclass

import torch
from transformers import PreTrainedTokenizerBase
from typing import Union, Optional, Type, TypeVar, List, Dict, Any

from transformers.file_utils import PaddingStrategy

from ibformers.data.collators.collators.base import BaseCollator

T = TypeVar("T")


def safe_dataclass_init(dataclass_: Type[T], **kwargs) -> T:
    valid_field_names = [f.name for f in fields(dataclass_)]
    valid_kwargs = {k: v for k, v in kwargs.items() if k in valid_field_names}
    return dataclass_(**valid_kwargs)


class UniversalDataCollator:
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 max_length: Optional[int] = None,
                 pad_to_multiple_of: Optional[int] = None, **kwargs):

        available_collators = BaseCollator.extra_collators
        self.base_collator = BaseCollator(
            tokenizer=tokenizer,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of
        )
        self.collators: List[BaseCollator] = [
            safe_dataclass_init(
                collator,
                tokenizer=tokenizer,
                padding=padding,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                **kwargs
            )
            for collator in available_collators
        ]

        self.field_to_collator: Dict[str, List[BaseCollator]] = defaultdict(list)

        for collator in self.collators:
            for field in collator.padded_fields:
                self.field_to_collator[field].append(collator)

    def __call__(self, features: [List[Dict[str, Any]]]):
        to_collate = set(features[0].keys())
        collated = self.base_collator(features)
        target_length = len(collated[self.base_collator.padded_fields[0]][0])
        to_collate -= collated.keys()
        while len(to_collate) != 0:
            field = to_collate.pop()
            extra_collator = self.field_to_collator[field][0]
            extra_collated = extra_collator(features, target_length)
            to_collate -= extra_collated.keys()
            collated = dict(**collated, **extra_collated)

        # TODO: custom types for fields
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in collated.items()}
        return batch
