import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel


class BertForNERC(nn.Module):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, bert_model, num_labels=2):
        super().__init__()
        self.num_labels = num_labels

        self.bert: BertModel = bert_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = self.classifier(outputs[0])
        return logits
