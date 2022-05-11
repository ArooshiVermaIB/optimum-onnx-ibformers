import logging
import logging
import os
from typing import Union, Optional, Callable

import torch
from torch import nn
from transformers import PreTrainedModel, WEIGHTS_NAME, PretrainedConfig
from transformers.file_utils import PushToHubMixin
from transformers.modeling_utils import unwrap_model, get_parameter_dtype

logger = logging.getLogger(__name__)


class PreTrainedLikeMixin(PreTrainedModel):
    base_model_prefix = ""
    config_class = PretrainedConfig
    _keys_to_ignore_on_save = None
    _keys_to_ignore_on_load_missing = None
    _keys_to_ignore_on_load_unexpected = None

    def tie_weights(self):
        pass

    # TODO: ask Bartosz why we need separate save_pretrained method

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        save_config: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        **kwargs,
    ):
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo = self._create_or_get_repo(save_directory, **kwargs)

        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = unwrap_model(self)

        # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5
        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.torch_dtype = str(dtype).split(".")[1]

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # Save the config
        if save_config:
            model_to_save.config.save_pretrained(save_directory)

        # Save the model
        if state_dict is None:
            state_dict = model_to_save.state_dict()

        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None:
            state_dict = {k: v for k, v in state_dict.items() if k not in self._keys_to_ignore_on_save}

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        save_function(state_dict, output_model_file)

        logger.info(f"Model weights saved in {output_model_file}")

    def retrieve_modules_from_names(self, names, add_prefix=False, remove_prefix=False):
        module_keys = set([".".join(key.split(".")[:-1]) for key in names])

        # torch.nn.ParameterList is a special case where two parameter keywords
        # are appended to the module name, *e.g.* bert.special_embeddings.0
        module_keys = module_keys.union(set([".".join(key.split(".")[:-2]) for key in names if key[-1].isdigit()]))

        retrieved_modules = []
        # retrieve all modules that has at least one missing weight name
        for name, module in self.named_modules():
            if remove_prefix:
                name = ".".join(name.split(".")[1:]) if name.startswith(self.base_model_prefix) else name
            elif add_prefix:
                name = ".".join([self.base_model_prefix, name]) if len(name) > 0 else self.base_model_prefix

            if name in module_keys:
                retrieved_modules.append(module)

        return retrieved_modules

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        pretrained_func = PreTrainedModel.from_pretrained.__func__
        return pretrained_func(cls, pretrained_model_name_or_path, *model_args, **kwargs)

    @classmethod
    def _set_default_torch_dtype(cls, dtype: torch.dtype) -> torch.dtype:
        pretrained_func = PreTrainedModel._set_default_torch_dtype.__func__
        return pretrained_func(cls, dtype)

    @classmethod
    def _load_state_dict_into_model(
        cls, model, state_dict, pretrained_model_name_or_path, ignore_mismatched_sizes=False, _fast_init=True
    ):
        pretrained_func = PreTrainedModel._load_state_dict_into_model.__func__
        return pretrained_func(
            cls, model, state_dict, pretrained_model_name_or_path, ignore_mismatched_sizes, _fast_init
        )
