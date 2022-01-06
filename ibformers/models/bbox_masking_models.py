import torch
from torch import nn
from torch.nn import CrossEntropyLoss
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

        self.indx_to_classifier_mapping = {
            0: self.xl_classifier,
            1: self.yl_classifier,
            2: self.xh_classifier,
            3: self.yh_classifier,
        }

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
        for bbox_idx, bbox_classifier in self.indx_to_classifier_mapping.items():
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
