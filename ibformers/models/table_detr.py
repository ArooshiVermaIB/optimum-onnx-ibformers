import json
import logging
import os

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Union, Dict, Any, Optional, Callable, List, Tuple, NamedTuple

from torchvision.ops import roi_pool
from transformers import CONFIG_NAME, PretrainedConfig, PreTrainedModel, WEIGHTS_NAME
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import unwrap_model, get_parameter_dtype

from ibformers.datasets.table_utils import DETR_DETECTION_CLASS_THRESHOLDS, DetrDetectionClassNames
from ibformers.third_party.detr.models import build_model
import ibformers.third_party.detr.datasets.transforms as R
from ibformers.third_party.detr.util.misc import NestedTensor
from ibformers.utils.pretrained_model_mixin import PreTrainedLikeMixin

logger = logging.getLogger(__name__)


class DetrConfig:
    """
    Detr config. Defaults to detection config.
    """

    def __init__(self, **kwargs):
        self.backbone = kwargs.get("backbone", "resnet18")
        # do not use pretrained backbone as weights will be replaced by from_pretrained
        self.pretrained_backbone = kwargs.get("pretrained_backbone", False)
        self.num_classes = kwargs.get("num_classes", 2)
        self.dilation = kwargs.get("dilation", False)
        self.position_embedding = kwargs.get("position_embedding", "sine")
        self.emphasized_weights = kwargs.get("emphasized_weights", {})
        self.enc_layers = kwargs.get("enc_layers", 6)
        self.dec_layers = kwargs.get("dec_layers", 6)
        self.dim_feedforward = kwargs.get("dim_feedforward", 2048)
        self.hidden_dim = kwargs.get("hidden_dim", 256)
        self.dropout = kwargs.get("dropout", 0.1)
        self.nheads = kwargs.get("nheads", 8)
        self.num_queries = kwargs.get("num_queries", 15)
        self.pre_norm = kwargs.get("pre_norm", True)
        self.masks = kwargs.get("masks", False)
        self.aux_loss = kwargs.get("aux_loss", False)
        self.mask_loss_coef = kwargs.get("mask_loss_coef", 1)
        self.dice_loss_coef = kwargs.get("dice_loss_coef", 1)
        self.ce_loss_coef = kwargs.get("ce_loss_coef", 1)
        self.bbox_loss_coef = kwargs.get("bbox_loss_coef", 5)
        self.giou_loss_coef = kwargs.get("giou_loss_coef", 2)
        self.eos_coef = kwargs.get("eos_coef", 0.4)
        self.set_cost_class = kwargs.get("set_cost_class", 1)
        self.set_cost_bbox = kwargs.get("set_cost_bbox", 5)
        self.set_cost_giou = kwargs.get("set_cost_giou", 2)
        self.table_detection_threshold = kwargs.get(
            "table_detection_threshold", DETR_DETECTION_CLASS_THRESHOLDS[DetrDetectionClassNames.TABLE]
        )
        self.extra_head_num_classes = kwargs.get("extra_head_num_classes", 0)
        self.extra_loss_coef = kwargs.get("extra_loss_coef", 1)


class CombinedTableDetrConfig(PretrainedConfig):
    def __init__(self, detection_config_data: Dict[str, Any], structure_config_data: Dict[str, Any], **kwargs):
        # hacky - super's init not called, we only want the serialization methods
        self.detection_config_data = DetrConfig(**detection_config_data).__dict__
        self.structure_config_data = DetrConfig(**structure_config_data).__dict__
        self.table_label2id = kwargs.get("table_label2id")
        self.table_id2label = kwargs.get("table_id2label")

    @property
    def detection_config(self):
        return DetrConfig(**self.detection_config_data)

    @property
    def structure_config(self):
        return DetrConfig(**self.structure_config_data)

    def to_json_string(self, use_diff: bool = False) -> str:
        """Overriden with no diff option"""
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"


DetrPrediction = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
DetectionModelOutput = Tuple[Dict[str, torch.FloatTensor], List[DetrPrediction]]
StructureModelOutput = Tuple[Dict[str, torch.FloatTensor], List[List[DetrPrediction]]]
TableDetrOutput = Tuple[torch.FloatTensor, List[Tuple[DetrPrediction, List[DetrPrediction]]]]


class CombinedTableDetrModel(PreTrainedLikeMixin):
    base_model_prefix = "table-detr"
    config_class = CombinedTableDetrConfig

    def __init__(self, config: CombinedTableDetrConfig, **kwargs):
        super().__init__(config)

        self.detection_config = self.config.detection_config
        self.structure_config = self.config.structure_config

        if hasattr(config, "table_id2label"):
            self.detection_config.extra_head_num_classes = len(config.table_label2id)
        # TODO: add a config option to add additional classifier which will be applied on the transformer outputs
        self.detection_model, self.detection_criterion, self.detection_postprocessors = build_model(
            self.detection_config
        )
        self.structure_model, self.structure_criterion, self.structure_postprocessors = build_model(
            self.structure_config
        )

        self.num_table_labels = len(config.table_label2id)
        self.table_classfier = nn.Linear(config.detection_config.hidden_dim, self.num_table_labels)

        self.normalize = R.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.inference_mode = False

    def inference(self, mode: bool):
        self.inference_mode = mode

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        page_images,
        table_page_bbox,
        structure_images=None,
        structure_image_size=None,
        page_margins=None,
        detection_boxes=None,
        detection_labels=None,
        table_label_ids=None,
        structure_boxes=None,
        structure_labels=None,
        images=None,
        record_table_page_no=None,
        structure_image_bbox=None,
        bbox_shift_vector=None,
        **kwargs
    ):

        detection_losses, detection_results = self.forward_detection(
            page_images, table_page_bbox, detection_boxes, detection_labels, table_label_ids
        )

        # pure prediction case
        if self.inference_mode:
            structure_losses, structure_results = self.forward_structure_manual(
                page_images, detection_results, page_margins
            )
        # training/evaluation
        else:
            structure_losses, structure_results = self.forward_structure_input(
                structure_images=structure_images,
                structure_image_size=structure_image_size,
                structure_boxes=structure_boxes,
                structure_labels=structure_labels,
                bbox_shift_vector=bbox_shift_vector,
            )

        total_detection_loss = sum(
            detection_losses[k] * self.detection_criterion.weight_dict[k]
            for k in detection_losses.keys()
            if k in self.detection_criterion.weight_dict
        )
        total_structure_loss = sum(
            structure_losses[k] * self.structure_criterion.weight_dict[k]
            for k in structure_losses.keys()
            if k in self.structure_criterion.weight_dict
        )
        loss = total_detection_loss + total_structure_loss
        results = [(d_result, s_result) for d_result, s_result in zip(detection_results, structure_results)]
        return loss, results

    def get_results_from_extra_head(self, out_logits):
        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)

        return (scores, labels)

    def forward_detection(
        self, page_images: NestedTensor, table_page_bbox, detection_boxes=None, detection_labels=None,
            table_label_ids=None, **kwargs
    ) -> DetectionModelOutput:
        image_sizes = table_page_bbox[:, [3, 2]]
        mask = torch.ones_like(page_images.mask, dtype=page_images.mask.dtype)
        for i, data in enumerate(image_sizes):
            mask[i, : data[0], : data[1]] = False
        normalized_images, _ = self.normalize(page_images.tensors.to(get_parameter_dtype(self.detection_model)))
        images = NestedTensor(normalized_images * ~mask.unsqueeze(1), mask)
        outputs = self.detection_model(images)
        results = self.detection_postprocessors["bbox"](outputs, image_sizes)
        results = [(r["scores"], r["boxes"], r["labels"]) for r in results]

        # add extra label
        if self.detection_config.extra_head_num_classes:
            extra_results = self.get_results_from_extra_head(outputs['pred_extra'])
            # results += extra_results

            results = [r + (score,) + (label,) for r, score, label in zip(results, extra_results[0], extra_results[1])]

        if detection_boxes is not None:
            targets = [
                {"labels": labels, "boxes": boxes.float(), "extra_labels": extra_labels}
                for labels, boxes, extra_labels in zip(detection_labels, detection_boxes, table_label_ids)
            ]
            loss = self.detection_criterion(outputs, targets)
            return loss, results
        return {}, results

    def forward_structure_manual(self, page_images, detection_output, page_margins, **kwargs) -> StructureModelOutput:
        batch_results = []
        tensors, masks = page_images.decompose()
        with torch.no_grad():
            for image, mask, single_output, page_margin in zip(tensors, masks, detection_output, page_margins):
                image_results: List[DetrPrediction] = []
                valid_table_indexes = (single_output[0] > self.detection_config.table_detection_threshold) & (
                    single_output[2] == 0
                )
                detected_table_bboxes = single_output[1][valid_table_indexes]
                detected_labels = single_output[4][valid_table_indexes]

                for bbox, tab_lab in zip(detected_table_bboxes, detected_labels):
                    input_image, image_size, shift_back_tensor = self.prepare_structure_input(
                        image, mask, bbox, page_margin
                    )
                    structure_output = self.structure_model(input_image)
                    results = self.structure_postprocessors["bbox"](structure_output, image_size.to(bbox.device))[0]
                    results["boxes"] += shift_back_tensor
                    results_list = (results["scores"], results["boxes"], results["labels"], tab_lab)
                    image_results.append(results_list)
                batch_results.append(image_results)
        return {}, batch_results

    def prepare_structure_input(self, image: Tensor, mask: Tensor, bbox: Tensor, page_margin: Tensor):
        expanded = bbox + bbox.new_tensor(
            [
                -page_margin,
                -page_margin,
                page_margin,
                page_margin,
            ]
        )
        subimage_bbox = torch.clamp(expanded, min=0).long()
        # note the reversed bbox coordinates for image subindexing
        new_image = image[None, :, subimage_bbox[1] : subimage_bbox[3], subimage_bbox[0] : subimage_bbox[2]]
        new_mask = mask[None, subimage_bbox[1] : subimage_bbox[3], subimage_bbox[0] : subimage_bbox[2]]
        normalized_image, _ = self.normalize(new_image)
        new_image = NestedTensor(normalized_image, new_mask)
        image_size = torch.as_tensor(normalized_image.size()[2:])
        shift_back_tensor = bbox.new_tensor([subimage_bbox[0], subimage_bbox[1], subimage_bbox[0], subimage_bbox[1]])
        return new_image, image_size.unsqueeze(0), shift_back_tensor

    def forward_structure_input(
        self, structure_images, structure_image_size, structure_boxes, structure_labels, bbox_shift_vector
    ) -> StructureModelOutput:
        normalized_images, _ = self.normalize(structure_images.tensors)
        images = NestedTensor(normalized_images, structure_images.mask)
        outputs = self.structure_model(images)
        results = self.structure_postprocessors["bbox"](outputs, structure_image_size[:, :2])
        results = [[(r["scores"], r["boxes"] - bbox_shift_vector[i], r["labels"])] for i, r in enumerate(results)]
        if structure_boxes is not None:
            targets = [
                {"labels": labels, "boxes": boxes.float()} for labels, boxes in zip(structure_labels, structure_boxes)
            ]
            loss = self.structure_criterion(outputs, targets)
            return loss, results
        return {}, results
