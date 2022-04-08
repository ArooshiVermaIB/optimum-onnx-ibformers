from dataclasses import dataclass
from typing import ClassVar, List, Optional

import torch

from ibformers.data.collators.collators.base import BaseCollator, CollatorABC
from ibformers.third_party.detr.util.misc import nested_tensor_from_tensor_list


@dataclass
class ImageCollator(BaseCollator):
    _supported_fields: ClassVar[List[str]] = ["image", "images"]

    @property
    def supported_fields(self) -> List[str]:
        return self._supported_fields

    def _collate_features(self, features, target_length: Optional[int] = None):
        feature_keys = self._get_feature_keys(features)
        assert any(
            f in feature_keys for f in self.supported_fields
        ), f"Neither of {self.supported_fields} columns was found in the inputs"
        present_supported_names = [key for key in feature_keys if key in self.supported_fields]
        batch = {}
        for feature_name in present_supported_names:
            assert all(
                feature[feature_name].shape[0] == 1 for feature in features
            ), "Collator supports only single pages"
            feature_batch = [feature[feature_name][0] for feature in features]
            batch[feature_name] = feature_batch
        return batch


@dataclass
class DetrSubImageImageExtractor(CollatorABC):
    _supported_fields: ClassVar[List[str]] = ["table_page_no", "structure_image_bbox", "images"]

    @property
    def supported_fields(self) -> List[str]:
        return self._supported_fields

    def _collate_features(self, features, target_length: Optional[int] = None):
        feature_keys = list(features[0].keys())
        assert all(
            f in feature_keys for f in self.supported_fields
        ), f"Neither of {self.supported_fields} columns was found in the inputs"
        page_images = []
        table_images = []
        for feature in features:
            subimage_bbox = feature["structure_image_bbox"]
            page_image = feature["images"][feature["table_page_no"]]
            table_image = page_image[:, subimage_bbox[1] : subimage_bbox[3], subimage_bbox[0] : subimage_bbox[2]]
            page_images.append(page_image)
            table_images.append(table_image)

        page_images = [torch.tensor(image) for image in page_images]
        table_images = [torch.tensor(image) for image in table_images]
        page_images = nested_tensor_from_tensor_list(page_images)
        table_images = nested_tensor_from_tensor_list(table_images)

        return {"page_images": page_images, "structure_images": table_images}
