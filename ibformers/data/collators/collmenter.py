# TODO: better name
from typing import Union, Optional, List, Type, Dict, Any, Callable

from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from transformers.data.data_collator import InputDataClass
from transformers.file_utils import PaddingStrategy

from ibformers.data.collators.augmenters.base import BaseAugmenter, AugmenterManager
from ibformers.data.collators.collators.universal import UniversalDataCollator


class CollatorWithAugmentation(Callable[[List[InputDataClass]], Dict[str, Any]]):
    """
    A joint class for feature collation and data augmentation.

    While it is not the most convenient to keep both those functionalities in single class,
    it fits nicely into transformer's Trainer.

    The initialization of this class is kinda tricky - we want to be able to define data processing
    pipeline with information on what augmenters to use, but the initialization itself has to be done later,
    in the training loop itself.
    Hence, the two step creation process. First, a custom subclass should be defined with information
    on what data collators should be used. For this, you can use `get_collator_class` function (see below).
    This creates a subclass with selected augmenters saved into `augmenters_to_use` class attribute. Then,
    the instance is initialized in the training loop.
    """
    augmenters_to_use: List[Type[BaseAugmenter]]

    def __init__(self, tokenizer: PreTrainedTokenizerBase,
                 model: PreTrainedModel,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 max_length: Optional[int] = None,
                 pad_to_multiple_of: Optional[int] = None,
                 augmenter_kwargs: Optional[Dict] = None):
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
        self.collator = UniversalDataCollator(
            tokenizer=tokenizer,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of
        )
        augmenter_kwargs = {} if augmenter_kwargs is None else augmenter_kwargs
        self.augmenter = AugmenterManager(
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
