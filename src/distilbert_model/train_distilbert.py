import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

CSV_PATH = "labels_merged_new.csv"
MODEL_NAME = "distilbert-base-uncased"
OUT_DIR = "bert_circuit__new_model"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------- 1. Dataset wrapper --------
class TextLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# -------- 2. Load data --------
def load_data():
    df = pd.read_csv(CSV_PATH)
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    return train_df, val_df


# -------- 3. Training loop --------
def train():
    train_df, val_df = load_data()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "non_circuit", 1: "circuit"},
        label2id={"non_circuit": 0, "circuit": 1},
    )
    model.to(DEVICE)

    train_dataset = TextLabelDataset(train_df["text"], train_df["label"], tokenizer)
    val_dataset = TextLabelDataset(val_df["text"], val_df["label"], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # class weights for imbalance (more weight on class 1)
    labels_np = train_df["label"].values
    pos_weight = (len(labels_np) - labels_np.sum()) / max(labels_np.sum(), 1)
    class_weights = torch.tensor([1.0, float(pos_weight)], device=DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 4
    total_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")

            optimizer.zero_grad()
            outputs = model(**batch)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ---- Validate ----
        model.eval()
        all_labels = []
        all_preds = []
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                labels = batch.pop("labels")
                outputs = model(**batch)
                logits = outputs.logits
                loss = loss_fn(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, pos_label=1)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"acc={acc:.3f} | f1_circuit={f1:.3f}"
        )

    # save
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("Saved trained model to", OUT_DIR)


if __name__ == "__main__":
    train()
