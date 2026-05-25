import pandas as pd
import torch
import joblib
import os
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not os.path.exists("dataset.csv"):
    print("Ошибка: dataset.csv не найден. Запустите сначала create_database.py")
    exit()

print("Загрузка токенизатора")
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

print("Загрузка данных")
df = pd.read_csv("dataset.csv")
intents = sorted(df["intent"].unique())
label2id = {label: idx for idx, label in enumerate(intents)}
id2label = {idx: label for label, idx in label2id.items()}
df["label"] = df["intent"].map(label2id)

print(f"Интенты: {intents}")
print(f"Примеров: {len(df)}")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

def tokenize(texts):
    return tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=64
    )

train_encodings = tokenize(train_texts)
val_encodings = tokenize(val_texts)

class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = IntentDataset(train_encodings, train_labels)
val_dataset = IntentDataset(val_encodings, val_labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }

print("Загрузка модели...")
model = AutoModelForSequenceClassification.from_pretrained(
    "DeepPavlov/rubert-base-cased",
    num_labels=len(intents),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
    problem_type="single_label_classification"
)

print("Настройка обучения...")
training_args = TrainingArguments(
    output_dir="./bert_intent_model",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=False,
    report_to="none",
    logging_steps=10,
    save_total_limit=1,
)

data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

print("Начало обучения")
trainer.train()

print("Оценка")
results = trainer.evaluate()
print(f"Accuracy: {results['eval_accuracy']:.4f}")
print(f"F1-score: {results['eval_f1']:.4f}")

print("Сохранение модели")
trainer.save_model("./bert_intent_model")
tokenizer.save_pretrained("./bert_intent_model")

metadata = {
    "label2id": label2id,
    "id2label": id2label,
    "intents": intents,
    "max_length": 64
}
joblib.dump(metadata, "./bert_intent_model/metadata.pkl")

print("Готово. Модель сохранена в ./bert_intent_model")