from datasets import Dataset

from factory import get_tokenizer, get_ne_classifier
from ne_classifier import BertForNERC
from scripts.anntools import Collection, Sentence
from pathlib import Path
import numpy as np
import json
from transformers import Trainer, TrainingArguments
import random
import torch
from transformers import BertTokenizerFast, BertForTokenClassification, BertConfig, BertModel
from datasets import load_metric

# class TokenDataset(Dataset):
#     def __init__(self, tokens, labels):
#         assert(len(tokens) == len(labels))
#         self.tokens = tokens
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.tokens)
#
#     def __getitem__(self, item):
#         return self.tokens[item], self.labels[item]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tokenizer = get_tokenizer()
    classifier = get_ne_classifier()

    with open('medline_train.json', 'r') as f:
        train_dataset = json.load(f)
    train_dataset = list(zip(train_dataset['tokens'], train_dataset['labels']))
    random.shuffle(train_dataset)
    val_dataset = train_dataset[int(0.9*len(train_dataset)):]
    train_dataset = train_dataset[:int(0.9*len(train_dataset))]

    train_tokens, train_labels = zip(*train_dataset)
    print(classifier(torch.Tensor([train_tokens[0]])))

    # print('Starting training')
    # metric = load_metric("accuracy")
    # trainer = Trainer(
    #     model=classifier,
    #     args=TrainingArguments('test_trainer'),
    #     train_dataset=TokenDataset(*zip(*train_dataset)),
    #     eval_dataset=TokenDataset(*zip(*val_dataset)),
    #     compute_metrics=lambda eval_prediction: metric.compute(predictions=np.argmax(eval_prediction[0], axis=-1),
    #                                                            references=eval_prediction[1])
    # )
    # print('Evaluating')
    # trainer.evaluate()