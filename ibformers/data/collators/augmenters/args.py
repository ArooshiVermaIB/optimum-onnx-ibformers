from dataclasses import fields, make_dataclass, Field, dataclass, field

from typing import Type, Tuple, List, Optional

from ibformers.data.collators.augmenters.base import BaseAugmenter


@dataclass
class BaseAugmenterArguments:
    """
    Contains arguments which are not specific to specific augmentators
    """
    augmenters_list: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of augmentators to use on input data. "
                    "If None the parameter will be obtained from pipeline defaults"
        },
    )


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
    params = []
    for fld in param_fields:
        # default value will be None so only if argument is passed it will be used to change the default
        fld.default = None
        params.append((fld.name, fld.type, fld))
    return params


AugmenterArguments = make_dataclass("AugmenterArguments", _get_augmenter_param_fields(),
                                    bases=(BaseAugmenterArguments,))
