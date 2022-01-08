from typing import Union

import numpy as np
import torch
from pathlib import Path

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel

from scripts.anntools import Collection, Sentence
import random


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {'input_ids': torch.tensor(self.encodings[idx]),
                'labels': torch.tensor(self.labels[idx])}

    def __len__(self):
        return len(self.labels)


def ne_label_token_sentence(sentence_tokens: list, sentence: Sentence, tokenizer) -> np.ndarray:
    ne_to_label_mapping = {
        'Action': 1,
        'Concept': 3,
        'Predicate': 5,
        'Reference': 7
    }
    # Tokenize each keyphrase for reference
    keyphrase_tokens = tokenizer([k.text for k in sentence.keyphrases])['input_ids']

    labels = np.zeros(len(sentence_tokens))
    for keyphrase, tokens in zip(sentence.keyphrases, keyphrase_tokens):
        s_pointer = 0
        actual_tokens = tokens[1:-1]
        # Iterate over sentence tokens trying to find the exact keyphrase sequence
        while s_pointer < len(sentence_tokens):
            if sentence_tokens[s_pointer:s_pointer+len(actual_tokens)] == actual_tokens:
                label_to_assign = ne_to_label_mapping[keyphrase.label]
                # First token in keyphrase sequence gets B label (begin)
                labels[s_pointer] = label_to_assign
                # The rest get I label (inside)
                for i in range(s_pointer+1, s_pointer+len(actual_tokens)):
                    labels[i] = label_to_assign+1
                break
            s_pointer += 1

    return labels


def tokenize_collection(c: Collection, tokenizer):
    tokens = []
    labels = []
    c_pointer = 0
    sentence_tokens = tokenizer([s.text for s in c.sentences], padding=True)['input_ids']
    for i in range(len(sentence_tokens)):
        tokens.append(sentence_tokens[i])
        labels.append(ne_label_token_sentence(sentence_tokens[i], c.sentences[c_pointer+i], tokenizer)
                      .astype(int)
                      .tolist())
    return tokens, labels


def generate_NERC_dataset(path: str, tokenizer_path: str):
    c: Collection = Collection().load(Path(path))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokens, labels = tokenize_collection(c, tokenizer)
    train_tokens, val_tokens, train_labels, val_labels = train_test_split(tokens, labels, test_size=0.2)
    return Dataset(train_tokens, train_labels), Dataset(val_tokens, val_labels)


def extract_hidden_states(tokens, model):
    with torch.no_grad():
        sentence_output = model(**tokens)
    return torch.stack(sentence_output.hidden_states).sum(0).squeeze()


def extract_relations(c: Collection, tokenizer, model, all_relations=False):
    relations = {x: (i+1)
                 for i, x in
                 enumerate(["is-a", "same-as", "part-of", "has-property", "causes",
                            "entails", "in-context", "in-place", "in-time",
                            "subject", "target", "domain", "arg",])
                 }
    dataset_x, dataset_y = [], []
    batch_size = 64
    c_pointer = 0
    while c_pointer < len(c):
        sentence_tokens = tokenizer(
            [s.text for s in c.sentences[c_pointer:c_pointer+batch_size]],
            padding=True,
            return_tensors="pt"
        )
        hidden_states = extract_hidden_states(sentence_tokens, model)

        for s_index, s in enumerate(c.sentences[c_pointer:c_pointer+batch_size]):
            keyphrase_tokens = {k.id: set() for k in s.keyphrases}
            for k in s.keyphrases:
                for span_m, span_M in k.spans:
                    for token_pos in range(sentence_tokens[s_index].char_to_token(span_m),
                                           sentence_tokens[s_index].char_to_token(span_M-1)+1):
                        keyphrase_tokens[k.id].add(token_pos)
            for r in s.relations:
                for origin_k in keyphrase_tokens[r.origin]:
                    for dest_k in keyphrase_tokens[r.destination]:
                        dataset_x.append(torch.cat((hidden_states[s_index][origin_k],
                                                    hidden_states[s_index][dest_k])))
                        dataset_y.append(relations[r.label])
            if all_relations:
                for k1 in keyphrase_tokens.keys():
                    for k2 in keyphrase_tokens.keys():
                        if k1 == k2:
                            continue
                        for origin_k in keyphrase_tokens[k1]:
                            for dest_k in keyphrase_tokens[k2]:
                                is_relation = False
                                for r in s.relations:
                                    if k1 == r.origin and k2 == r.destination:
                                        is_relation = True
                                        break
                                if not is_relation:
                                    dataset_x.append(torch.cat((hidden_states[s_index][origin_k],
                                                                hidden_states[s_index][dest_k])))
                                    dataset_y.append(0)
            else:
                label_0_counter = 0
                while label_0_counter < 3:
                    origin, dest = random.randrange(len(hidden_states[s_index])),\
                                   random.randrange(len(hidden_states[s_index]))
                    is_relation = False
                    for r in s.relations:
                        for origin_k in keyphrase_tokens[r.origin]:
                            for dest_k in keyphrase_tokens[r.destination]:
                                if origin_k == origin and dest_k == dest:
                                    is_relation = True
                                    break
                    if not is_relation:
                        dataset_x.append(torch.cat((hidden_states[s_index][origin],
                                                    hidden_states[s_index][dest])))
                        dataset_y.append(0)
                        label_0_counter += 1

        c_pointer += batch_size
    return dataset_x, dataset_y


def generate_RE_datasets(path: str, tokenizer_path: str, two_datasets=False, all_relations=False, label_0_ratio=1):
    c: Collection = Collection().load(Path(path))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModel.from_pretrained(tokenizer_path, output_hidden_states=True)
    embeddings, labels = extract_relations(c, tokenizer, model, all_relations)
    if two_datasets:
        # Prepare binary dataset
        labels_binary = [0 if x == 0 else 1 for x in labels]
        data_0 = list([(x, y) for x, y in zip(embeddings, labels_binary) if y == 0])
        random.shuffle(data_0)
        data_0 = data_0[:int(len([y for y in labels_binary if y == 1]) * label_0_ratio)]
        data_1 = list([(x, y) for x, y in zip(embeddings, labels_binary) if y == 1])
        data_0 += data_1
        embeddings_binary, labels_binary = zip(*data_0)
        train_embeddings_bin, val_embeddings_bin, train_labels_bin, val_labels_bin = \
            train_test_split(embeddings_binary, labels_binary, test_size=0.2)
        # Prepare relation dataset
        embeddings_with_relation, labels_with_relation = zip(*[(x, y) for x, y in zip(embeddings, labels) if y != 0])
        train_embeddings_rel, val_embeddings_rel, train_labels_rel, val_labels_rel = \
            train_test_split(embeddings_with_relation, labels_with_relation, test_size=0.2)
        return Dataset(train_embeddings_bin, train_labels_bin), Dataset(val_embeddings_bin, val_labels_bin), \
               Dataset(train_embeddings_rel, train_labels_rel), Dataset(val_embeddings_rel, val_labels_rel)
    else:
        if label_0_ratio is not None:
            data_0 = list([(x, y) for x, y in zip(embeddings, labels) if y == 0])
            random.shuffle(data_0)
            data_0 = data_0[:int(len([y for y in labels if y != 0]) * label_0_ratio)]
            data_1 = list([(x, y) for x, y in zip(embeddings, labels) if y != 0])
            data_0 += data_1
            embeddings, labels = zip(*data_0)
        train_embeddings, val_embeddings, train_labels, val_labels = \
            train_test_split(embeddings, labels, test_size=0.2)
        return Dataset(train_embeddings, train_labels), Dataset(val_embeddings, val_labels)


def generate_all_possible_relation_pairs(path: Union[str, Collection], tokenizer, model):
    if isinstance(path, Collection):
        test_c = path
    else:
        test_c = Collection().load(Path(path))
    dataset_x, sentences, keyphrases = [], [], []
    batch_size = 64
    c_pointer = 0
    while c_pointer < len(test_c):
        sentence_tokens = tokenizer(
            [s.text for s in test_c.sentences[c_pointer:c_pointer+batch_size]],
            padding=True,
            return_tensors="pt"
        )
        hidden_states = extract_hidden_states(sentence_tokens, model)
        for s_index, s in enumerate(test_c.sentences[c_pointer:c_pointer+batch_size]):
            keyphrase_tokens = {k.id: set() for k in s.keyphrases}
            for k in s.keyphrases:
                for span_m, span_M in k.spans:
                    for token_pos in range(sentence_tokens[s_index].char_to_token(span_m),
                                           sentence_tokens[s_index].char_to_token(span_M-1)+1):
                        keyphrase_tokens[k.id].add(token_pos)
            for k1 in keyphrase_tokens.keys():
                for k2 in keyphrase_tokens.keys():
                    if k1 == k2:
                        continue
                    for origin_k in keyphrase_tokens[k1]:
                        for dest_k in keyphrase_tokens[k2]:
                            dataset_x.append(torch.cat((hidden_states[s_index][origin_k],
                                                        hidden_states[s_index][dest_k])))
                            sentences.append(c_pointer+s_index)
                            keyphrases.append((k1, k2))

        c_pointer += batch_size
    return dataset_x, sentences, keyphrases


