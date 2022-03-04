import math

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import LayoutLMConfig, AutoModelForTokenClassification, AutoConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, MaskedLMOutput
from transformers.models.layoutlm.modeling_layoutlm import (
    LayoutLMLayerNorm,
    LayoutLMPreTrainedModel,
    LayoutLMEncoder,
    LayoutLMPooler,
    LayoutLMModel,
    LayoutLMOnlyMLMHead,
    LayoutLMForMaskedLM,
    LayoutLMForSequenceClassification,
    LayoutLMForTokenClassification,
)

from ibformers.models.bbox_masking_models import LayoutLMForMaskedLMAndLayoutRegression


class LayoutLMPositionlessConfig(LayoutLMConfig):
    model_type = "layoutlm_positionless"


class LayoutLMPositionlessEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(LayoutLMPositionlessEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = LayoutLMLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids=None,
        bbox=None,
        token_type_ids=None,
        inputs_embeds=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        words_embeddings = inputs_embeds
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The :obj:`bbox`coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (
            words_embeddings
            + left_position_embeddings
            + upper_position_embeddings
            + right_position_embeddings
            + lower_position_embeddings
            + h_position_embeddings
            + w_position_embeddings
            + token_type_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LayoutLMPositionlessModel(LayoutLMModel):
    config_class = LayoutLMPositionlessConfig

    def __init__(self, config):
        super(LayoutLMPositionlessModel, self).__init__(config)
        self.config = config
        self.embeddings = LayoutLMPositionlessEmbeddings(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Examples::

            >>> from transformers import LayoutLMTokenizer, LayoutLMModel
            >>> import torch

            >>> tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
            >>> model = LayoutLMModel.from_pretrained('microsoft/layoutlm-base-uncased')

            >>> words = ["Hello", "world"]
            >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

            >>> token_boxes = []
            >>> for word, box in zip(words, normalized_word_boxes):
            ...     word_tokens = tokenizer.tokenize(word)
            ...     token_boxes.extend([box] * len(word_tokens))
            >>> # add bounding boxes of cls + sep tokens
            >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

            >>> encoding = tokenizer(' '.join(words), return_tensors="pt")
            >>> input_ids = encoding["input_ids"]
            >>> attention_mask = encoding["attention_mask"]
            >>> token_type_ids = encoding["token_type_ids"]
            >>> bbox = torch.tensor([token_boxes])

            >>> outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)

            >>> last_hidden_states = outputs.last_hidden_state
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if bbox is None:
            bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids,
            bbox=bbox,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class LayoutLMPositionlessForMaskedLM(LayoutLMForMaskedLM):
    config_class = LayoutLMPositionlessConfig

    def __init__(self, config):
        super().__init__(config)

        self.layoutlm = LayoutLMPositionlessModel(config)


class LayoutLMPositionlessForSequenceClassification(LayoutLMForSequenceClassification):
    config_class = LayoutLMPositionlessConfig

    def __init__(self, config):
        super().__init__(config)
        self.layoutlm = LayoutLMPositionlessModel(config)


class LayoutLMPositionlessForTokenClassification(LayoutLMForTokenClassification):
    config_class = LayoutLMPositionlessConfig

    def __init__(self, config):
        super().__init__(config)
        self.layoutlm = LayoutLMPositionlessModel(config)


class LayoutLMPositionlessForMaskedLMAndLayoutRegression(LayoutLMForMaskedLMAndLayoutRegression):
    config_class = LayoutLMPositionlessConfig

    def __init__(self, config):
        super().__init__(config)
        self.layoutlm = LayoutLMPositionlessModel(config)


AutoConfig.register("layoutlm_positionless", LayoutLMPositionlessConfig)
AutoModelForTokenClassification.register(LayoutLMPositionlessConfig, LayoutLMPositionlessForTokenClassification)
