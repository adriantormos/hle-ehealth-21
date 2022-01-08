import torch
import transformers
from torch import nn
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, \
    TrainingArguments

from src.preprocessing import generate_NERC_dataset, generate_RE_datasets
from src.utils import compute_metrics

import pytorch_lightning as pl


# class RelationClassifierPL(pl.LightningModule):
#     def __init__(self, input=784*2, output=14):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input, 500),
#             nn.ReLU(),
#             nn.Linear(500, output),
#             nn.Softmax()
#         )
#
#     def forward(self, x):
#         return self.network(x)
#
#     def training_step(self, batch, batch_idx):
#         # training_step defined the train loop.
#         # It is independent of forward
#         x, y = batch
#         x = x.view(x.size(0), -1)
#         x = self.network(x)
#         loss = nn.functional.cross_entropy(x, y)
#         # Logging to TensorBoard by default
#         self.log("train_loss", loss)
#         return loss
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer


class RelationClassifier(nn.Module):
    def __init__(self, input=768*2, output=14):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input, 500),
            nn.ReLU(),
            nn.Linear(500, output)
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


def train_task_B1(pretrained_model: str, epochs=5, model_name='taskB_model.pt'):
    model_rel = RelationClassifier()
    train_dataset, val_dataset = generate_RE_datasets(
        '2021/ref/training/medline.1200.es.txt',
        pretrained_model
    )
    train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Training relation classifier
    for epoch in range(epochs):
        running_loss = 0.0
        optimizer = torch.optim.SGD(model_rel.parameters(), lr=0.001, momentum=0.9)
        correct, total = 0, 0
        for i, data in enumerate(train_dataset, 0):
            inputs, labels = data['input_ids'], data['labels']
            optimizer.zero_grad()
            outputs = model_rel(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Epoch {epoch}: train acc {correct/total}', end=' ')
        correct, total = 0, 0
        for i, data in enumerate(val_dataset, 0):
            with torch.no_grad():
                inputs, labels = data['input_ids'], data['labels']
                outputs = model_rel(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'val acc {correct/total}')
    torch.save(model_rel.state_dict(), f'./{model_name}')


def train_task_B2(pretrained_model: str, epochs=5):
    model_bin = RelationClassifier(output=1).to('cuda')
    model_rel = RelationClassifier().to('cuda')
    train_dataset_bin, val_dataset_bin, train_dataset_rel, val_dataset_rel = generate_RE_datasets(
        '2021/ref/training/medline.1200.es.txt',
        pretrained_model,
        two_datasets=True,
        all_relations=True
    )
    train_dataset = torch.utils.data.DataLoader(train_dataset_bin, batch_size=64, shuffle=True)
    val_dataset = torch.utils.data.DataLoader(val_dataset_bin, batch_size=64, shuffle=False)
    # Training binary classifier
    for epoch in range(epochs):
        running_loss = 0.0
        optimizer = torch.optim.SGD(model_bin.parameters(), lr=0.001, momentum=0.9)
        correct, total = 0, 0
        for i, data in enumerate(train_dataset, 0):
            inputs, labels = data['input_ids'], data['labels']
            inputs = inputs.to('cuda')
            optimizer.zero_grad()

            outputs = model_bin(inputs)
            labels = labels.to('cuda').unsqueeze(1).float()

            loss = nn.BCEWithLogitsLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            total += labels.size(0)
            correct += ((outputs > 0.5) == labels).sum().item()
        print(f'Epoch {epoch}: train acc {correct/total}', end=' ')
        correct, total = 0, 0
        for i, data in enumerate(val_dataset, 0):
            with torch.no_grad():
                inputs, labels = data['input_ids'], data['labels']
                inputs = inputs.to('cuda')

                outputs = model_bin(inputs)
                labels = labels.to('cuda').unsqueeze(1).float()

                total += labels.size(0)
                correct += ((torch.sigmoid(outputs) > 0.5) == labels).sum().item()
        print(f'val acc {correct/total}')
    torch.save(model_bin.state_dict(), './taskB_model_bin_es.pt')

    train_dataset = torch.utils.data.DataLoader(train_dataset_rel, batch_size=64, shuffle=True)
    val_dataset = torch.utils.data.DataLoader(val_dataset_rel, batch_size=64, shuffle=False)
    # Training relation classifier
    for epoch in range(epochs):
        running_loss = 0.0
        optimizer = torch.optim.SGD(model_rel.parameters(), lr=0.001, momentum=0.9)
        correct, total = 0, 0
        for i, data in enumerate(train_dataset, 0):
            inputs, labels = data['input_ids'], data['labels']
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            optimizer.zero_grad()

            outputs = model_rel(inputs)
            _, predicted = torch.max(outputs, 1)

            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Epoch {epoch}: train acc {correct/total}', end=' ')
        correct, total = 0, 0
        for i, data in enumerate(val_dataset, 0):
            with torch.no_grad():
                inputs, labels = data['input_ids'], data['labels']
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

                outputs = model_rel(inputs)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'val acc {correct/total}')
    torch.save(model_rel.state_dict(), './taskB_model_rel_es.pt')


    # trainer = pl.Trainer(max_epochs=10, gpus=1)
    # trainer.fit(model, train_dataset, val_dataset)
    # trainer.test(test_dataloaders=val_dataset)