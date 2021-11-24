from dataclasses import fields, make_dataclass, Field

from typing import Type, Tuple, List

from ibformers.data.collators.augmenters.base import BaseAugmenter


def _get_augmenter_param_fields() -> List[Tuple[str, Type, Field]]:
    """
    Get fields defined by augmenter dataclasses, but drop the ones from the BaseAugmenter.

    Returns:
        List of (field_name, field_type, field) tuples required by `make_dataclass`.
    """
    param_fields = set()
    for augmenter in BaseAugmenter.augmenters:
        param_fields.update(fields(augmenter))
    for base_field_name in fields(BaseAugmenter):
        param_fields.remove(base_field_name)
    return [(field.name, field.type, field) for field in param_fields]


AugmenterArguments = make_dataclass("AugmenterArguments", _get_augmenter_param_fields())
