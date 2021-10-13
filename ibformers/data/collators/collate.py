import random
from dataclasses import dataclass
from typing import Union, Optional

import torch
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
    DataCollatorForTokenClassification,
)
from transformers.file_utils import PaddingStrategy


@dataclass
class DataCollatorWithBBoxesForTokenClassification:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        feature_keys = list(features[0].keys())
        assert (
            "bboxes" in feature_keys or "bbox" in feature_keys
        ), "Bbox column not found in the inputs"
        bbox_name = "bbox" if "bbox" in feature_keys else "bboxes"
        label_name = "label" if "label" in feature_keys else "labels"
        labels = (
            [feature[label_name] for feature in features] if label_name in feature_keys else None
        )
        bboxes = [feature[bbox_name] for feature in features]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[bbox_name] = [
                bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in bboxes
            ]
            batch["labels"] = [
                label + [self.label_pad_token_id] * (sequence_length - len(label))
                for label in labels
            ]
            if "mqa_ids" in feature_keys:
                mqa_ids = [feature["mqa_ids"] for feature in features]
                batch["mqa_ids"] = [
                    mqa_id + [1] * (sequence_length - len(mqa_id)) for mqa_id in mqa_ids
                ]
        else:
            raise ValueError("Only right padding is supported")

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}

        # for k, v in batch.items():
        #     print(f'{k}:{v.shape}')

        return batch


@dataclass
class DataCollatorWithBBoxesAugmentedForTokenClassification(
    DataCollatorWithBBoxesForTokenClassification
):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    """

    def __call__(self, features):
        batch = super(DataCollatorWithBBoxesAugmentedForTokenClassification, self).__call__(
            features
        )

        if self.model.training:
            # do bbox augmentation
            bbox = batch["bbox"]
            non_zeros_idx = (bbox[:, :, 0] != 0).to(bbox.dtype)

            x_offset = random.randrange(-20, 20)
            y_offset = random.randrange(-20, 20)
            x_scale = random.uniform(0.95, 1.05)
            y_scale = random.uniform(0.95, 1.05)

            bbox[:, :, [0, 2]] = (
                (bbox[:, :, [0, 2]] * x_scale + x_offset).clamp(1, 999).to(dtype=bbox.dtype)
            )
            bbox[:, :, [1, 3]] = (
                (bbox[:, :, [1, 3]] * y_scale + y_offset).clamp(1, 999).to(dtype=bbox.dtype)
            )
            bbox = bbox * non_zeros_idx[:, :, None]
            batch["bbox"] = bbox

        return batch


def DataCollatorFor1DTokenClassification(*args, **kwargs):
    kwargs.pop("model")  # TODO: Make this more generic
    return DataCollatorForTokenClassification(*args, **kwargs)
