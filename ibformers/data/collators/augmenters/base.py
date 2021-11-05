from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass

import torch
from typing import Dict, ClassVar, List, Type

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ibformers.data.collators.collators.universal import safe_dataclass_init


class AugmenterABC(ABC):

    @abstractmethod
    def augment(self, batch: Dict[str, torch.Tensor]):
        pass


@dataclass
class BaseAugmenter(AugmenterABC, metaclass=ABCMeta):
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    augmenters: ClassVar[List["BaseAugmenter"]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.augmenters.append(cls)


class Augmenter(AugmenterABC):

    def __init__(self, tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel,
                 augmenters_to_use: List[Type[BaseAugmenter]], **augmenter_kwargs):
        self.tokenizer = tokenizer
        self.model = model

        self.augmenters = [
            safe_dataclass_init(augmenter, tokenizer=tokenizer, model=model, **augmenter_kwargs)
            for augmenter in augmenters_to_use
        ]

    def augment(self, batch: Dict[str, torch.Tensor]):
        for augmenter in self.augmenters:
            batch = augmenter.augment(batch)
        return batch
