from typing import Dict, Type

from ..ib_extraction.ib_extraction import IbExtraction
from ..ib_classification.ib_classification import IbClassificationDs
from ..ib_split_class.ib_split_class import IbSplitClass

IB_DATASETS: Dict[str, Type["IbDs"]] = {
    cls.CONFIG_NAME: cls for cls in [IbExtraction, IbClassificationDs, IbSplitClass]
}
