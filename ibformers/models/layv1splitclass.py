from typing import Optional, List
import torch
from transformers import LayoutLMConfig, LayoutLMModel, LayoutLMPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class SplitClassifierConfig(LayoutLMConfig):
    def __init__(
        self, class_weights: Optional[List[float]] = None, split_weights: Optional[List[float]] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.class_weights = class_weights if class_weights is not None else [1.0] * self.num_labels
        self.split_weights = split_weights if split_weights is not None else [1.0] * 2


class SplitClassifier(LayoutLMPreTrainedModel):
    config_class = SplitClassifierConfig

    def __init__(self, config):
        super(SplitClassifier, self).__init__(config)
        self.layoutlm = LayoutLMModel(config)
        self.dense1 = torch.nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(config.hidden_dropout_prob)
        self.dense2 = torch.nn.Linear(config.hidden_size, 2)
        self.cls_dense = torch.nn.Linear(config.hidden_size, self.config.num_labels)
        self.splitter_criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(self.config.split_weights))
        self.classifier_criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(self.config.class_weights))

        self.init_weights()

    def forward(
        self, input_ids, bboxes, attention_mask, next_input_ids, next_bboxes, next_attention_mask, sc_labels=None
    ):
        left_page = self.layoutlm(
            input_ids=input_ids,
            bbox=bboxes,
            attention_mask=attention_mask,
        )

        right_page = self.layoutlm(input_ids=next_input_ids, bbox=next_bboxes, attention_mask=next_attention_mask)

        both_page = torch.cat([left_page.pooler_output, right_page.pooler_output], dim=1)

        x = self.dense1(both_page)

        x = self.relu1(x)

        splitter_logits = self.dense2(x)

        drop_out_cls = self.drop1(left_page.pooler_output)

        classifier_logits = self.cls_dense(drop_out_cls)

        logits = torch.cat([splitter_logits, classifier_logits], dim=1)

        if sc_labels is not None:
            splitter_loss = self.splitter_criterion(splitter_logits, sc_labels[:, 0])

            classifier_loss = self.classifier_criterion(classifier_logits, sc_labels[:, 1])

            loss = splitter_loss + classifier_loss

            return SequenceClassifierOutput(loss=loss, logits=logits)
        else:
            return SequenceClassifierOutput(logits=logits)
