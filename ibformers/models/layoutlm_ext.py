import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import normalize
from transformers import LayoutLMv2PreTrainedModel
from transformers.modeling_outputs import (
    TokenClassifierOutput,
)
from transformers.models.layoutlm.modeling_layoutlm import (
    LayoutLMPreTrainedModel,
    LayoutLMModel,
)
from transformers.models.layoutlmv2 import LayoutLMv2Model

from ibformers.models.triplet_loss import TripletLoss


def calculate_clf_loss(loss_fct, logits, labels, attention_mask):
    num_classes = logits.size(-1)
    if attention_mask is not None:
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, num_classes)
        active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels))
        col_loss = loss_fct(active_logits, active_labels)
    else:
        col_loss = loss_fct(logits.view(-1, num_classes), labels.view(-1))
    return col_loss


class LayoutLMForTableStructureClassification(LayoutLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layoutlm = LayoutLMModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # TODO: unhardcode
        self.num_order_classes = 200  # 30 rows + non-row
        self.row_id_classifier = nn.Linear(config.hidden_size, self.num_order_classes)
        self.col_id_classifier = nn.Linear(config.hidden_size, self.num_order_classes)
        self.table_id_classifier = nn.Linear(config.hidden_size, self.num_order_classes)

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
        stacked_table_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
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
        token_row_ids, token_col_ids, token_table_ids = stacked_table_labels.unbind(dim=-1)
        sequence_output = self.dropout(sequence_output)
        row_logits = self.row_id_classifier(sequence_output)
        col_logits = self.col_id_classifier(sequence_output)
        table_logits = self.table_id_classifier(sequence_output)

        loss = None
        loss_fct = CrossEntropyLoss()

        if token_row_ids is not None:

            row_loss = calculate_clf_loss(loss_fct, row_logits, token_row_ids, attention_mask)
            if loss is None:
                loss = row_loss
            else:
                loss += row_loss

        if token_col_ids is not None:

            col_loss = calculate_clf_loss(loss_fct, col_logits, token_col_ids, attention_mask)
            if loss is None:
                loss = col_loss
            else:
                loss += col_loss

        if token_table_ids is not None:

            table_loss = calculate_clf_loss(loss_fct, table_logits, token_table_ids, attention_mask)
            if loss is None:
                loss = table_loss
            else:
                loss += table_loss

        logits = torch.cat((row_logits, col_logits, table_logits), -1)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LayoutLMForCheckeredTableTokenClassification(LayoutLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlm = LayoutLMModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_order_classes = 3

        # create checkered classifier which will classify each column and row into one of the two classes
        # neibouring classes will have different classes and that's how we would now about the separation
        # so columns in the 4 column label will be classified as follows [1,2,1,2]
        # all tokens outside the table will get 0 class
        self.row_classifier = nn.Linear(config.hidden_size, self.num_order_classes)
        self.col_classifier = nn.Linear(config.hidden_size, self.num_order_classes)

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
        chattered_row_ids=None,
        chattered_col_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
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

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        row_logits = self.row_classifier(sequence_output)
        col_logits = self.col_classifier(sequence_output)

        loss = None
        cls_loss = None
        group_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                cls_loss = loss_fct(active_logits, active_labels)
            else:
                cls_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = cls_loss

        if chattered_row_ids is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = row_logits.view(-1, self.num_order_classes)
                active_labels = torch.where(
                    active_loss,
                    chattered_row_ids.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(chattered_row_ids),
                )
                row_loss = loss_fct(active_logits, active_labels)
            else:
                row_loss = loss_fct(row_logits.view(-1, self.num_order_classes), chattered_row_ids.view(-1))
            if loss is None:
                loss = row_loss
            else:
                loss += row_loss

        if chattered_col_ids is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = col_logits.view(-1, self.num_order_classes)
                active_labels = torch.where(
                    active_loss,
                    chattered_col_ids.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(chattered_col_ids),
                )
                col_loss = loss_fct(active_logits, active_labels)
            else:
                col_loss = loss_fct(col_logits.view(-1, self.num_order_classes), chattered_col_ids.view(-1))
            if loss is None:
                loss = col_loss
            else:
                loss += col_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        logits = torch.cat((logits, row_logits, col_logits), -1)
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LayoutLMv2ForCheckeredTableTokenClassification(LayoutLMv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_order_classes = 3

        # create checkered classifier which will classify each column and row into one of the two classes
        # neibouring classes will have different classes and that's how we would now about the separation
        # so columns in the 4 column label will be classified as follows [1,2,1,2]
        # all tokens outside the table will get 0 class
        self.row_classifier = nn.Linear(config.hidden_size, self.num_order_classes)
        self.col_classifier = nn.Linear(config.hidden_size, self.num_order_classes)

        self.init_weights()

    def get_input_embeddings(self):
        return self.layoutlmv2.embeddings.word_embeddings

    def forward(
        self,
        input_ids=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        chattered_row_ids=None,
        chattered_col_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        # only take the text part of the output representations
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        row_logits = self.row_classifier(sequence_output)
        col_logits = self.col_classifier(sequence_output)

        loss = None
        cls_loss = None
        group_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                cls_loss = loss_fct(active_logits, active_labels)
            else:
                cls_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = cls_loss

        if chattered_row_ids is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = row_logits.view(-1, self.num_order_classes)
                active_labels = torch.where(
                    active_loss,
                    chattered_row_ids.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(chattered_row_ids),
                )
                row_loss = loss_fct(active_logits, active_labels)
            else:
                row_loss = loss_fct(row_logits.view(-1, self.num_order_classes), chattered_row_ids.view(-1))
            if loss is None:
                loss = row_loss
            else:
                loss += row_loss

        if chattered_col_ids is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = col_logits.view(-1, self.num_order_classes)
                active_labels = torch.where(
                    active_loss,
                    chattered_col_ids.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(chattered_col_ids),
                )
                col_loss = loss_fct(active_logits, active_labels)
            else:
                col_loss = loss_fct(col_logits.view(-1, self.num_order_classes), chattered_col_ids.view(-1))
            if loss is None:
                loss = col_loss
            else:
                loss += col_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        logits = torch.cat((logits, row_logits, col_logits), -1)
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LayoutLMForTokenClassificationWithGroupClustering(LayoutLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlm = LayoutLMModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.hidden_dim_size = 512
        self.row_embedder = nn.Linear(config.hidden_size, self.hidden_dim_size)
        self.col_embedder = nn.Linear(config.hidden_size, self.hidden_dim_size)  # TODO: unhardcode

        # self.order_id_classifier = nn.Sequential(
        #     nn.Linear(config.hidden_size, self.hidden_dim_size),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim_size, self.num_order_classes),
        # )

        self.group_loss_weight = config.group_loss_weight

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
        token_row_ids=None,
        token_col_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
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

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        row_embeddings = self.row_embedder(sequence_output)
        row_embeddings = normalize(row_embeddings, p=2.0)
        col_embeddings = self.col_embedder(sequence_output)
        col_embeddings = normalize(col_embeddings, p=2.0)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                cls_loss = loss_fct(active_logits, active_labels)
            else:
                cls_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # loss = cls_loss

        if token_row_ids is not None:

            triplet_loss_fct = TripletLoss(self.device, 1.0)
            group_losses = []
            for batch_group_ids, batch_group_logits in zip(token_row_ids, row_embeddings):
                valid_example_ids = torch.where(batch_group_ids >= 0)[0]
                valid_logits = batch_group_logits[valid_example_ids]
                valid_labels = batch_group_ids[valid_example_ids]
                group_losses.append(triplet_loss_fct(valid_logits, valid_labels))
            row_triplet_loss = torch.mean(torch.stack(group_losses))
            if loss is None:
                loss = row_triplet_loss * self.group_loss_weight
            else:
                loss += row_triplet_loss * self.group_loss_weight

        if token_col_ids is not None:

            triplet_loss_fct = TripletLoss(self.device, 1.0)
            group_losses = []
            for batch_group_ids, batch_group_logits in zip(token_col_ids, col_embeddings):
                valid_example_ids = torch.where(batch_group_ids >= 0)[0]
                valid_logits = batch_group_logits[valid_example_ids]
                valid_labels = batch_group_ids[valid_example_ids]
                group_losses.append(triplet_loss_fct(valid_logits, valid_labels))
            col_triplet_loss = torch.mean(torch.stack(group_losses))
            if loss is None:
                loss = col_triplet_loss * self.group_loss_weight
            else:
                loss += col_triplet_loss * self.group_loss_weight

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        logits = torch.cat((logits, row_embeddings, col_embeddings), -1)
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SelfAttentionMask(nn.Module):
    def __init__(self, hidden_size: int, qkv_size: int):
        super().__init__()

        self.query = nn.Linear(hidden_size, 2 * qkv_size)
        self.key = nn.Linear(hidden_size, 2 * qkv_size)
        self.value = nn.Linear(hidden_size, 2 * qkv_size)

        self.attention_head_size = qkv_size
        self.num_attention_heads = 2

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
    ):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)
        return attention_scores


def compute_adjacency_loss(loss_fct, logits, labels, attention_mask):
    row_adjancency_labels = (labels[:, :, None] == labels[:, None, :]).long()
    ignore_idxs = labels == -100
    row_adjancency_labels[ignore_idxs, :] = -100
    row_adjancency_labels.transpose(1, 2)[ignore_idxs, :] = -100

    return loss_fct(logits, row_adjancency_labels)


class LayoutLMForTableAdjacencyMatrix(LayoutLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlm = LayoutLMModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.hidden_dim_size = 768

        self.row_grouper = SelfAttentionMask(config.hidden_size, self.hidden_dim_size)
        self.col_grouper = SelfAttentionMask(config.hidden_size, self.hidden_dim_size)
        self.table_grouper = SelfAttentionMask(config.hidden_size, self.hidden_dim_size)
        self.table_classifier = nn.Linear(config.hidden_size, 2)
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
        stacked_table_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
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
        token_row_ids, token_col_ids, token_table_ids = stacked_table_labels.unbind(dim=-1)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        row_adjacency_matrix = self.row_grouper(sequence_output)
        col_adjacency_matrix = self.col_grouper(sequence_output)
        table_adjacency_matrix = self.table_grouper(sequence_output)
        table_classifier_output = self.table_classifier(sequence_output)
        loss = None

        loss_fct = CrossEntropyLoss(weight=torch.tensor([1.0, 1.0], device=self.device))
        if token_row_ids is not None:
            row_adj_loss = compute_adjacency_loss(loss_fct, row_adjacency_matrix, token_row_ids, attention_mask)
            if loss is None:
                loss = row_adj_loss
            else:
                loss += row_adj_loss

        if token_col_ids is not None:
            col_adj_loss = compute_adjacency_loss(loss_fct, col_adjacency_matrix, token_col_ids, attention_mask)
            if loss is None:
                loss = col_adj_loss
            else:
                loss += col_adj_loss

        if token_table_ids is not None:
            table_adj_loss = compute_adjacency_loss(loss_fct, table_adjacency_matrix, token_table_ids, attention_mask)

            clf_loss_fct = CrossEntropyLoss()
            table_clf_loss = calculate_clf_loss(
                clf_loss_fct, table_classifier_output, token_table_ids.clamp(max=1), attention_mask
            )
            if loss is None:
                loss = table_adj_loss + table_clf_loss
            else:
                loss += table_adj_loss + table_clf_loss

        row_adjacency_logits = row_adjacency_matrix[:, 1, :, :] - row_adjacency_matrix[:, 0, :, :]
        col_adjacency_logits = col_adjacency_matrix[:, 1, :, :] - col_adjacency_matrix[:, 0, :, :]
        table_adjacency_logits = table_adjacency_matrix[:, 1, :, :] - table_adjacency_matrix[:, 0, :, :]
        logits = torch.cat((row_adjacency_logits, col_adjacency_logits, table_adjacency_logits), -1)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
