import torch
from typing import Any, Optional, Union

from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from torchvision import transforms

from typing import NewType

InputDataClass = NewType("InputDataClass", Any)


class DataCollatorForKeyValueExtraction:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, str, PaddingStrategy] = "max_length",
        max_length: Optional[int] = 512,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        **kwargs,
    ):
        """
        Data collator that will dynamically pad the inputs received, as well as the labels.
        Args:
            tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
                The tokenizer used for encoding the data.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`max_length`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                among:
                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the returned list and optionally padding length (see above). defaults to 512.
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
                7.5 (Volta).
            label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
                The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        """
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        patch_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))]
        )
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        images = None
        if "images" in features[0]:
            images = torch.stack(
                [patch_transform(d.pop("images")[0][::-1, :, :].transpose(1, 2, 0).copy()) for d in features],
                dim=0,
            )
            # images = torch.stack([torch.tensor(d.pop("images")) for d in features])
            IMAGE_LEN = int(images.shape[-1] / 16) * int(images.shape[-1] / 16) + 1

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if images is not None:
            batch["images"] = images
            batch = {
                k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) and k == "attention_mask" else v
                for k, v in batch.items()
            }
            visual_attention_mask = torch.ones((len(batch["input_ids"]), IMAGE_LEN), dtype=torch.long)
            batch["attention_mask"] = torch.cat([batch["attention_mask"], visual_attention_mask], dim=1)

        if labels is None:
            return batch

        has_bbox_input = "bbox" in features[0]
        has_position_input = "position_ids" in features[0]
        padding_idx = self.tokenizer.pad_token_id
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
            if has_bbox_input:
                # batch["bbox"] = [bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in batch["bbox"]]
                batch["bbox"] = [
                    bbox.tolist() + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in batch["bbox"]
                ]
            if has_position_input:
                batch["position_ids"] = [
                    position_id + [padding_idx] * (sequence_length - len(position_id))
                    for position_id in batch["position_ids"]
                ]

        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
            if has_bbox_input:
                batch["bbox"] = [[[0, 0, 0, 0]] * (sequence_length - len(bbox)) + bbox for bbox in batch["bbox"]]
            if has_position_input:
                batch["position_ids"] = [
                    [padding_idx] * (sequence_length - len(position_id)) + position_id
                    for position_id in batch["position_ids"]
                ]

        batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}

        if images is not None:
            visual_labels = torch.ones((len(batch["input_ids"]), IMAGE_LEN), dtype=torch.long) * -100
            batch["labels"] = torch.cat([batch["labels"], visual_labels], dim=1)

        return batch
