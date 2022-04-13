from collections import OrderedDict

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    LayoutLMConfig,
    PretrainedConfig,
    CONFIG_NAME,
)
from transformers.models.auto.auto_factory import _LazyAutoMapping, _BaseAutoModelClass, auto_class_update
from transformers.models.auto.configuration_auto import _LazyConfigMapping

from ibformers.models.layoutlm_positionless import (
    LayoutLMPositionlessConfig,
    LayoutLMPositionlessForTokenClassification,
    LayoutLMPositionlessForSequenceClassification,
    PositionlessSplitClassifier,
    PositionlessSplitClassifierConfig,
)
from ibformers.models.layv1splitclass import SplitClassifier, SplitClassifierConfig

MODEL_FOR_SPLIT_CLASSIFICATION_MAPPING = _LazyAutoMapping(OrderedDict([]), OrderedDict([]))

AutoConfig.register("layoutlm_positionless", LayoutLMPositionlessConfig)
AutoModelForTokenClassification.register(LayoutLMPositionlessConfig, LayoutLMPositionlessForTokenClassification)
AutoModelForSequenceClassification.register(LayoutLMPositionlessConfig, LayoutLMPositionlessForSequenceClassification)


class CustomAutoConfig:
    """
    Custom AutoConfig. Create custom version by subclassing and changing the config mapping.
    """

    CONFIG_MAPPING = _LazyConfigMapping({})

    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def for_model(cls, model_type: str, *args, **kwargs):
        if model_type in cls.CONFIG_MAPPING:
            config_class = cls.CONFIG_MAPPING[model_type]
            return config_class(*args, **kwargs)
        raise ValueError(
            f"Unrecognized model identifier: {model_type}. Should contain one of {', '.join(cls.CONFIG_MAPPING.keys())}"
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        kwargs["_from_auto"] = True
        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "model_type" in config_dict:
            config_class = cls.CONFIG_MAPPING[config_dict["model_type"]]
            return config_class.from_dict(config_dict, **kwargs)
        else:
            # Fallback: use pattern matching on the string.
            for pattern, config_class in cls.CONFIG_MAPPING.items():
                if pattern in str(pretrained_model_name_or_path):
                    return config_class.from_dict(config_dict, **kwargs)

        raise ValueError(
            f"Unrecognized model in {pretrained_model_name_or_path}. "
            f"Should have a `model_type` key in its {CONFIG_NAME}, or contain one of the following strings "
            f"in its name: {', '.join(cls.CONFIG_MAPPING.keys())}"
        )

    @classmethod
    def register(cls, model_type, config):
        if issubclass(config, PretrainedConfig) and config.model_type != model_type:
            raise ValueError(
                "The config you are passing has a `model_type` attribute that is not consistent with the model type "
                f"you passed (config has {config.model_type} and you passed {model_type}. Fix one of those so they "
                "match!"
            )
        cls.CONFIG_MAPPING.register(model_type, config)


# we need to instantiate it as empty and use the `register` to add new items. The default one are expected
# to exist in `transformers` library.
SPLIT_CLASSIFIER_MAPPING = _LazyConfigMapping({})
SPLIT_CLASSIFIER_MAPPING.register("layoutlm", SplitClassifierConfig)
SPLIT_CLASSIFIER_MAPPING.register("layoutlm_positionless", PositionlessSplitClassifierConfig)


class SplitClassifierAutoConfig(CustomAutoConfig):
    CONFIG_MAPPING = SPLIT_CLASSIFIER_MAPPING


class AutoModelForSplitClassification(_BaseAutoModelClass):
    config_class = SplitClassifierAutoConfig
    _model_mapping = MODEL_FOR_SPLIT_CLASSIFICATION_MAPPING


AutoModelForSplitClassification = auto_class_update(AutoModelForSplitClassification, head_doc="split classification")

AutoModelForSplitClassification.register(SplitClassifierConfig, SplitClassifier)
AutoModelForSplitClassification.register(PositionlessSplitClassifierConfig, PositionlessSplitClassifier)
