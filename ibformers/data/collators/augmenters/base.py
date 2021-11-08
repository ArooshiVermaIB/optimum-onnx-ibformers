from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass, make_dataclass, fields

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
    """
    Base class for augmenters.
    """
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    augmenters: ClassVar[List["BaseAugmenter"]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.augmenters.append(cls)


class AugmenterManager(AugmenterABC):
    """
    Handles augmentation using multiple augmenters.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel,
                 augmenters_to_use: List[Type[BaseAugmenter]], **augmenter_kwargs):
        """

        Args:
            tokenizer: Tokenizer for the model that the data is augmented for.
            model: Model for the training task
            augmenters_to_use: List of BaseAugmenter subclasses that will be used during the augmentation.
            The order of augmenters in this list is also the order that they will be called in.
            **augmenter_kwargs: Keyword arguments to the augmenters.
        """
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
