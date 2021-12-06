from collections import defaultdict
from dataclasses import fields, dataclass

import torch
from transformers import PreTrainedTokenizerBase, DataCollator
from typing import Union, Optional, Type, TypeVar, List, Dict, Any

from transformers.file_utils import PaddingStrategy

from ibformers.data.collators.collators.base import BaseCollator

T = TypeVar("T")


def safe_dataclass_init(dataclass_: Type[T], **kwargs) -> T:
    """
    Initialize DataClass with given kwargs, discarding the unnecessary ones.

    This is somewhat a workaround over us using dataclasses for classes that
    are not really holding any data. It discards any kwargs that are not present in
    Args:
        dataclass_: The dataclass to be initialized
        **kwargs: keyword arguments

    Returns:
        Initialized dataclass.

    """
    valid_field_names = [f.name for f in fields(dataclass_)]
    valid_kwargs = {k: v for k, v in kwargs.items() if k in valid_field_names}
    return dataclass_(**valid_kwargs)


class UniversalDataCollator:
    """
    Universal data collator. Collates every input feature using collators registered in
    BaseCollator.

    The collating process is done as follows:
    1. Use tokenizer-based BaseCollator to collate some of the features, depending on the tokenizer.
    2. For each remaining feature, look for its matching collator and use it to process the feature.
    Collate each extra feature to the same length as the first batch from BaseCollator.
    3. If some feature doesn't have a matching collator, throw an error.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            tokenizer: Tokenizer used for base collation
            padding: Options for padding
            max_length: Maximum length of the collated batch
            pad_to_multiple_of: If set, the collated batch will have dimensions that are a multiple
            of provided value. Useful for fp16 training.
            **kwargs: Extra arguments passed to additional collators.
        """

        available_collators = BaseCollator.extra_collators
        self.base_collator = BaseCollator(
            tokenizer=tokenizer,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        self.collators: List[BaseCollator] = [
            safe_dataclass_init(
                collator,
                tokenizer=tokenizer,
                padding=padding,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                **kwargs,
            )
            for collator in available_collators
        ]

        self.field_to_collator: Dict[str, List[BaseCollator]] = defaultdict(list)

        for collator in self.collators:
            for field in collator.supported_fields:
                self.field_to_collator[field].append(collator)

    @property
    def supported_fields(self) -> List[str]:
        return self.base_collator.supported_fields + list(self.field_to_collator.keys())

    def _get_collator_for_field(self, field_name: str) -> BaseCollator:
        collators = self.field_to_collator[field_name]
        if len(collators) > 1:
            raise RuntimeError(f"There are more than one collators for given filed.")
        try:
            return self.field_to_collator[field_name][0]
        except KeyError:
            raise KeyError(
                f"Could not find a collator for field {field_name}. "
                f"Make sure that the field name is correct, and define a subclass of "
                f"BaseCollator if it is not yet covered."
            )

    def __call__(self, features: [List[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        """
        Collate features into a batch.

        Calls BaseCollator first, and then extra collators for each remaining feature.

        Args:
            features: List of features

        Returns:
            Batch of collated tensors.

        Raises:
            KeyError: When a field name is present in features, but has no defined collator.
            RuntimeError: When a field name has multiple collators available.

        """
        to_collate = set(features[0].keys())
        collated = self.base_collator(features)
        target_length = len(collated[self.base_collator.supported_fields[0]][0])
        to_collate -= collated.keys()
        while len(to_collate) != 0:
            field = to_collate.pop()
            extra_collator = self._get_collator_for_field(field)
            extra_collated = extra_collator(features, target_length)
            to_collate -= extra_collated.keys()
            collated = dict(**collated, **extra_collated)

        # TODO: custom types for fields
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in collated.items()}
        return batch
