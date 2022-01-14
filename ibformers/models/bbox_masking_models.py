from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from transformers import LayoutLMConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.layoutlm import LayoutLMForMaskedLM


class LayoutLMForMaskedLMAndLayout(LayoutLMForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        self.xl_classifier = nn.Linear(config.hidden_size, config.max_2d_position_embeddings)
        self.yl_classifier = nn.Linear(config.hidden_size, config.max_2d_position_embeddings)
        self.xh_classifier = nn.Linear(config.hidden_size, config.max_2d_position_embeddings)
        self.yh_classifier = nn.Linear(config.hidden_size, config.max_2d_position_embeddings)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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
        labels=None,
        bbox_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        masked_lm_output = super().forward(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        sequence_output = masked_lm_output.hidden_states[-1]
        sequence_output = self.dropout(sequence_output)

        losses = []
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        for bbox_idx, bbox_classifier in (
            (0, self.xl_classifier),
            (1, self.yl_classifier),
            (2, self.xh_classifier),
            (3, self.yh_classifier),
        ):
            logits = bbox_classifier(sequence_output)
            labels = bbox_labels[:, :, bbox_idx]
            loss = loss_fct(logits.view(-1, 1024), labels.view(-1))
            losses.append(loss)

        loss = masked_lm_output.loss + torch.mean(torch.tensor(losses))

        return TokenClassifierOutput(
            loss=loss,
            logits=masked_lm_output.logits,
            hidden_states=masked_lm_output.hidden_states,
            attentions=masked_lm_output.attentions,
        )


class LayoutLMForBboxMaskingRegressionConfig(LayoutLMConfig):
    def __init__(self, bbox_scale_factor: Optional[float] = 500.0, smooth_loss_beta: Optional[float] = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.bbox_scale_factor = bbox_scale_factor if bbox_scale_factor is not None else 500.0
        self.smooth_loss_beta = smooth_loss_beta if smooth_loss_beta is not None else 1.0


class LayoutLMForMaskedLMAndLayoutRegression(LayoutLMForMaskedLM):
    config_class = LayoutLMForBboxMaskingRegressionConfig

    def __init__(self, config):
        super().__init__(config)

        self.bbox_scale_factor = config.bbox_scale_factor
        self.smooth_loss_beta = config.smooth_loss_beta

        self.bbox_regressor = nn.Linear(config.hidden_size, 4)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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
        labels=None,
        bbox_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        masked_lm_output = super().forward(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        sequence_output = masked_lm_output.hidden_states[-1]
        sequence_output = self.dropout(sequence_output)

        bbox_output = self.bbox_regressor(sequence_output)
        active_loss_idxs = (bbox_labels != -100).any(-1)

        active_predictions = bbox_output[active_loss_idxs] / self.bbox_scale_factor
        active_labels = bbox_labels[active_loss_idxs] / self.bbox_scale_factor
        loss_fct = SmoothL1Loss(beta=self.smooth_loss_beta)

        bbox_loss = loss_fct(active_predictions.view(-1), active_labels.view(-1))

        loss = masked_lm_output.loss + bbox_loss

        return TokenClassifierOutput(
            loss=loss,
            logits=masked_lm_output.logits,
            hidden_states=masked_lm_output.hidden_states,
            attentions=masked_lm_output.attentions,
        )
