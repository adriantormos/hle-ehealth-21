from pathlib import Path

from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer

from scripts.anntools import Collection
from src.preprocessing import Dataset
from src.postprocessing import extract_named_entities
from src.utils import compute_metrics


def pipeline_task_A(model_path: str) -> Collection:

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=9)
    test_c = Collection().load(Path('/content/2021/eval/testing/scenario2-taskA/input.txt'))
    test_input = tokenizer([s.text for s in test_c.sentences], padding=True)

    trainer = Trainer(
        model,
        TrainingArguments(
            '',
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
                                                     [[0] * len(test_input['input_ids'][i]) for i, _ in enumerate(test_input['input_ids'])]
                                                     ))
    sentences = extract_named_entities(tokenizer, predictions, test_input, test_c)
    return Collection(sentences)
