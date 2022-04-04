from dataclasses import make_dataclass, Field, field
import inspect
from typing import Type, Tuple, List, Optional

from .pipeline import PIPELINES


IGNORE_ARGS = ["tokenizer", "kwargs"]


def _get_preprocess_param_fields() -> List[Tuple[str, Type, Field]]:
    """
    Get fields in functions which are part of the preprocess step

    Returns:
        List of (field_name, field_type, field) tuples required by `make_dataclass`.
    """
    fn_list = list(set([fn for pl in PIPELINES.values() for fn in pl["preprocess"]]))
    ignore_args = IGNORE_ARGS + [list(inspect.signature(fn).parameters)[0] for fn in fn_list]
    params = [dict(inspect.signature(fn).parameters) for fn in fn_list]
    seen = {}
    fields = []
    for fn_params in params:
        for par_name, par in fn_params.items():
            if par_name in ignore_args:
                continue
            if par.annotation is inspect._empty:
                raise TypeError(f"Parameter {par_name} is not typed")
            if par_name in seen:
                continue
            seen[par_name] = True
            # default value will be None so only if argument is passed it will be used to change the default
            fld = field(default=None)
            fld.name = par_name
            fld.type = Optional[par.annotation]
            fields.append((fld.name, fld.type, fld))

    return fields


PreprocessArguments = make_dataclass("PreprocessArguments", _get_preprocess_param_fields())
