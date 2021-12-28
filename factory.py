
from ne_classifier import BertForNERC
from transformers import BertTokenizerFast, BertModel


def get_tokenizer():
    return BertTokenizerFast.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')


def get_ne_classifier():
    bert_model = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased'
                                           # , config=BertConfig(num_labels=5)
                                           )
    ne_classifier = BertForNERC(bert_model, num_labels=5)
    return ne_classifier