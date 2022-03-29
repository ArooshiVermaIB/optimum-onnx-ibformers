from collections import OrderedDict

from transformers import AutoConfig, AutoModelForTokenClassification, AutoModelForSequenceClassification, LayoutLMConfig
from transformers.models.auto.auto_factory import _LazyAutoMapping, _BaseAutoModelClass, auto_class_update

from ibformers.models.layoutlm_positionless import (
    LayoutLMPositionlessConfig,
    LayoutLMPositionlessForTokenClassification,
    LayoutLMPositionlessForSequenceClassification,
    PositionlessSplitClassifier,
)
from ibformers.models.layv1splitclass import SplitClassifier

MODEL_FOR_SPLIT_CLASSIFICATION_MAPPING = _LazyAutoMapping(OrderedDict([]), OrderedDict([]))

AutoConfig.register("layoutlm_positionless", LayoutLMPositionlessConfig)
AutoModelForTokenClassification.register(LayoutLMPositionlessConfig, LayoutLMPositionlessForTokenClassification)
AutoModelForSequenceClassification.register(LayoutLMPositionlessConfig, LayoutLMPositionlessForSequenceClassification)


class AutoModelForSplitClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SPLIT_CLASSIFICATION_MAPPING


AutoModelForSplitClassification = auto_class_update(AutoModelForSplitClassification, head_doc="split classification")

AutoModelForSplitClassification.register(LayoutLMConfig, SplitClassifier)
AutoModelForSplitClassification.register(LayoutLMPositionlessConfig, PositionlessSplitClassifier)
