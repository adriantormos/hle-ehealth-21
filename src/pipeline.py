from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer

from scripts.anntools import Collection
from src.preprocessing import Dataset, generate_all_possible_relation_pairs
from src.postprocessing import extract_named_entities, extract_relations
from src.utils import compute_metrics
from src.train import RelationClassifier


def pipeline_task_a(model_path: str, dataset_path: str) -> Collection:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=9)
    test_c = Collection().load(Path(dataset_path))
    test_input = tokenizer([s.text for s in test_c.sentences], padding=True)

    trainer = Trainer(
        model,
        TrainingArguments(
            '.',
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            num_train_epochs=4,
            weight_decay=0.01
        ),
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    predictions, labels, _ = trainer.predict(Dataset(test_input['input_ids'],
                                                     [[0] * len(test_input['input_ids'][i]) for i, _ in
                                                      enumerate(test_input['input_ids'])]
                                                     ))
    sentences = extract_named_entities(tokenizer, predictions, test_input, test_c)
    return Collection(sentences)


def pipeline_task_b1(model_path: str, preparator_path: str, dataset_path: str) -> Collection:
    model = RelationClassifier().to('cuda')
    model.load_state_dict(torch.load(model_path))
    tokenizer = AutoTokenizer.from_pretrained(preparator_path)
    preparator = AutoModelForTokenClassification.from_pretrained(preparator_path, num_labels=9,
                                                                 output_hidden_states=True)
    dataset_x, sentences, keyphrases = generate_all_possible_relation_pairs(dataset_path, tokenizer, preparator)
    batch_size = 64
    c_pointer = 0
    test_c = Collection().load(Path(dataset_path))
    while c_pointer < len(dataset_x):
        with torch.no_grad():
            inputs = torch.stack(dataset_x[c_pointer:c_pointer + batch_size]).to('cuda')
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            extract_relations(predicted,
                              sentences[c_pointer:c_pointer + batch_size],
                              keyphrases[c_pointer:c_pointer + batch_size],
                              test_c)
        c_pointer += batch_size
    return test_c


def pipeline_task_b2(model_bin_path: str, model_rel_path: str, preparator_path: str, dataset_path: str) -> Collection:
    model_bin = RelationClassifier(output=1).to('cuda')
    model_rel = RelationClassifier().to('cuda')
    model_bin.load_state_dict(torch.load(model_bin_path))
    model_rel.load_state_dict(torch.load(model_rel_path))
    tokenizer = AutoTokenizer.from_pretrained(preparator_path)
    preparator = AutoModelForTokenClassification.from_pretrained(preparator_path, num_labels=9,
                                                                 output_hidden_states=True)
    dataset_x, sentences, keyphrases = generate_all_possible_relation_pairs(dataset_path, tokenizer, preparator)
    batch_size = 64
    c_pointer = 0
    test_c = Collection().load(Path(dataset_path))
    while c_pointer < len(dataset_x):
        with torch.no_grad():
            inputs = torch.stack(dataset_x[c_pointer:c_pointer + batch_size]).to('cuda')

            outputs_bin = model_bin(inputs)
            predicted_bin = torch.sigmoid(outputs_bin) > 0.5
            possible_relation_data = [(i + c_pointer, x)
                                      for i, (x, y) in enumerate(zip(inputs, predicted_bin))
                                      if y == 1]
            if len(possible_relation_data) > 0:
                possible_relation_indices, possible_relation_inputs = zip(*possible_relation_data)
                possible_relation_inputs = torch.stack(possible_relation_inputs).to('cuda')

                outputs_rel = model_rel(possible_relation_inputs)
                _, predicted = torch.max(outputs_rel, 1)
                extract_relations(predicted,
                                  [sentences[i] for i in possible_relation_indices],
                                  [keyphrases[i] for i in possible_relation_indices],
                                  test_c)
        c_pointer += batch_size
    return test_c


def pipeline(nerc_model_path: str, re_paths: List[str], preparator_path: str, dataset_path: str) -> Collection:
    collection = Collection().load(Path(dataset_path))
    collection = pipeline_task_a(nerc_model_path, dataset_path)
    if len(re_paths) == 1:
        return pipeline_task_b1(re_paths[0], preparator_path, dataset_path)
    else:
        return pipeline_task_b2(re_paths[0], re_paths[1], preparator_path, dataset_path)
