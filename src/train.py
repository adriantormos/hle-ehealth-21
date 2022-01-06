import torch
import transformers
from torch import nn
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, \
    TrainingArguments

from src.preprocessing import generate_NERC_dataset, generate_RE_dataset
from src.utils import compute_metrics

import pytorch_lightning as pl


class RelationClassifierPL(pl.LightningModule):
    def __init__(self, input=784*2, output=14):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input, 500),
            nn.ReLU(),
            nn.Linear(500, output),
            nn.Softmax()
        )

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        x = self.network(x)
        loss = nn.functional.cross_entropy(x, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class RelationClassifier(nn.Module):
    def __init__(self, input=784*2, output=14):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input, 500),
            nn.ReLU(),
            nn.Linear(500, output),
            nn.Softmax()
        )

    def forward(self, x):
        return self.network(x)


def train_task_A(pretrained_model: str):

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = AutoModelForTokenClassification.from_pretrained(pretrained_model, num_labels=9)
    train_dataset, val_dataset = generate_NERC_dataset(
        '2021/ref/training/medline.1200.es.txt',
        pretrained_model
    )
    model_name = pretrained_model.split("/")[-1]
    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = transformers.Trainer(
        model,
        TrainingArguments(
            f"{model_name}-finetuned-ehealth21",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            num_train_epochs=4,
            weight_decay=0.01
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model()


def train_task_B(pretrained_model: str):
    model = RelationClassifier()
    train_dataset, val_dataset = generate_RE_dataset(
        '2021/ref/training/medline.1200.es.txt',
        pretrained_model
    )
    train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    for epoch in range(5):
        running_loss = 0.0
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        for i, data in enumerate(train_dataset, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        correct, total = 0, 0
        for i, data in enumerate(val_dataset, 0):
            with torch.no_grad():
                inputs, labels = data
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Epoch {epoch}: val acc {correct/total}')


    # trainer = pl.Trainer(max_epochs=10, gpus=1)
    # trainer.fit(model, train_dataset, val_dataset)
    # trainer.test(test_dataloaders=val_dataset)