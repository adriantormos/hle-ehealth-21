from ne_classifier import BertForNERC
from scripts.anntools import Collection, Sentence
from pathlib import Path
import numpy as np
import json

import torch
from transformers import BertTokenizerFast, BertForTokenClassification, BertConfig, BertModel


def get_tokenizer():
    return BertTokenizerFast.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')


def get_ne_classifier():
    bert_model = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased'
                                           # , config=BertConfig(num_labels=5)
                                           )
    ne_classifier = BertForNERC(bert_model, num_labels=5)
    return ne_classifier


def ne_label_token_sentence(sentence_tokens: list, sentence: Sentence, tokenizer: BertTokenizerFast) -> np.ndarray:
    ne_to_label_mapping = {
        'Action': 1,
        'Concept': 2,
        'Predicate': 3,
        'Reference': 4
    }
    keyphrase_tokens = tokenizer([k.text for k in sentence.keyphrases])['input_ids']
    labels = np.zeros(len(sentence_tokens))
    for keyphrase, tokens in zip(sentence.keyphrases, keyphrase_tokens):
        s_pointer = 0
        actual_tokens = tokens[1:-1]
        while s_pointer < len(sentence_tokens):
            if sentence_tokens[s_pointer:s_pointer+len(actual_tokens)] == actual_tokens:
                label_to_assign = ne_to_label_mapping[keyphrase.label]
                for i in range(s_pointer, s_pointer+len(actual_tokens)):
                    labels[i] = label_to_assign
                break
            s_pointer += 1

    return labels


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    c: Collection = Collection().load(Path("2021/ref/training/medline.1200.es.txt"))

    tokens = []
    labels = []
    tokenizer = get_tokenizer()
    batch_size = 64
    c_pointer = 0
    while c_pointer < len(c):
        sentence_tokens = tokenizer([s.text for s in c.sentences[c_pointer:c_pointer+batch_size]])['input_ids']
        for i in range(len(sentence_tokens)):
            tokens.append(sentence_tokens[i])
            labels.append(ne_label_token_sentence(sentence_tokens[i], c.sentences[c_pointer+i], tokenizer)
                          .astype(int)
                          .tolist())
        c_pointer += batch_size

    with open('medline_train.json', 'x') as f:
        json.dump({'tokens': tokens, 'labels': labels}, f, indent=4)