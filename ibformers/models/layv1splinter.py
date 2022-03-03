import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.layoutlm.modeling_layoutlm import (
    LayoutLMPreTrainedModel,
    LayoutLMModel,
)
from transformers.models.splinter.modeling_splinter import SplinterFullyConnectedLayer


SPLINTER_MAX_QUESTIONS = 20


class QuestionAwareTokenSelectionHead(nn.Module):
    """
    Implementation of Question-Aware Token Selection (QASS) head, similar to the one described in Splinter's paper
    Instead of selecting single span - we modified that to select all tokens which contribute to the answer
    This way we are not dependent on reading order and we could select entities in multiple places in document
    """

    def __init__(self, config):
        super().__init__()

        self.query_start_transform = SplinterFullyConnectedLayer(config.hidden_size, config.hidden_size)
        self.start_transform = SplinterFullyConnectedLayer(config.hidden_size, config.hidden_size)
        self.start_classifier = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.max_questions = SPLINTER_MAX_QUESTIONS

    def forward(self, inputs, positions):
        _, _, dim = inputs.size()
        positions = positions[:, : self.max_questions]
        index = positions.unsqueeze(-1).repeat(1, 1, dim)  # [batch_size, num_positions, dim]
        gathered_reps = torch.gather(inputs, dim=1, index=index)  # [batch_size, num_positions, dim]

        query_start_reps = self.query_start_transform(gathered_reps)  # [batch_size, num_positions, dim]
        start_reps = self.start_transform(inputs)  # [batch_size, seq_length, dim]

        hidden_states = self.start_classifier(query_start_reps)  # [batch_size, num_positions, dim]
        start_reps = start_reps.permute(0, 2, 1)  # [batch_size, dim, seq_length]
        ans_logits = torch.matmul(hidden_states, start_reps)

        return ans_logits


class LayoutSplinterModel(LayoutLMPreTrainedModel):
    """
    Architecture is using LayoutLM architecture and is doing token classification. In contrast to the standard
    token classification this one is question aware (or entity_name aware)
    """

    def __init__(self, config):
        super(LayoutSplinterModel, self).__init__(config)
        self.config = config

        self.layoutlm = LayoutLMModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.splinter_qass = QuestionAwareTokenSelectionHead(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        answer_token_label_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        question_positions=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        bin_logits = self.splinter_qass(sequence_output, question_positions).permute(0, 2, 1).contiguous()

        loss = None
        if answer_token_label_ids is not None:
            # TODO: change that into hyperparameter
            loss_fct = BCEWithLogitsLoss(pos_weight=torch.FloatTensor([10.0]).to(answer_token_label_ids.device))

            flat_bin_logits = bin_logits.view(-1)
            flat_ans_ids = answer_token_label_ids.view(-1)
            not_ignore_mask = flat_ans_ids != -100
            filter_bin_logits = flat_bin_logits[not_ignore_mask]
            filter_ans_ids = flat_ans_ids[not_ignore_mask].to(torch.float32)

            loss = loss_fct(filter_bin_logits, filter_ans_ids)

        if not return_dict:
            output = (bin_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=bin_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
