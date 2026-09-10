"""
train_pipeline.py
Script huan luyen mo hinh OneIE JointModel (XLM-RoBERTa + 3 classifier heads)
cho bai toan Vietnamese Political Event Extraction.
Giai quyet van de mat can bang lop (Class Imbalance) bang ham mat mat co trong so
theo dung cong thuc tu Pipeline_Joint_Learning.ipynb:
L = 0.4 * L_trigger + 0.3 * L_arg + 0.3 * L_event
"""

import os
import sys
import ast
import json
import time
import pickle
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizerFast
import pandas as pd
import numpy as np

# Reconfigure stdout/stderr for UTF-8
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# Thiet lap duong dan import
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from models.model import JointModel
except ImportError:
    from model import JointModel


class FastEventDataset(Dataset):
    """
    Dataset chuan bi du lieu phuc vu huan luyen OneIE JointModel.
    Can chinh subwords ve tung token tu goc va gan nhan the BIO.
    """

    def __init__(self, data: pd.DataFrame, tokenizer, trigger_encoder, arg_encoder, event_encoder, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.trigger_encoder = trigger_encoder
        self.arg_encoder = arg_encoder
        self.event_encoder = event_encoder
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = row["tokens"]
        trigger_tags = row["bio_tags"]
        arg_tags = row["argument_tags"]
        min_len = min(len(tokens), len(trigger_tags), len(arg_tags))
        tokens, trigger_tags, arg_tags = tokens[:min_len], trigger_tags[:min_len], arg_tags[:min_len]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        word_ids = encoding.word_ids()

        trigger_labels, arg_labels = [], []
        for word_idx in word_ids:
            if word_idx is None or word_idx >= len(trigger_tags):
                trigger_labels.append(-100)
                arg_labels.append(-100)
            else:
                t_tag = trigger_tags[word_idx]
                a_tag = arg_tags[word_idx]
                t_id = self.trigger_encoder.transform([t_tag])[0] if t_tag in self.trigger_encoder.classes_ else -100
                a_id = self.arg_encoder.transform([a_tag])[0] if a_tag in self.arg_encoder.classes_ else -100
                trigger_labels.append(t_id)
                arg_labels.append(a_id)

        e_id = self.event_encoder.transform([row["event_type"]])[0]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "trigger_labels": torch.tensor(trigger_labels, dtype=torch.long),
            "arg_labels": torch.tensor(arg_labels, dtype=torch.long),
            "event_label": torch.tensor(e_id, dtype=torch.long),
        }


def calculate_class_weights(labels: List[int], num_classes: int) -> torch.FloatTensor:
    """
    Tinh trong so phat nghich dao theo so luong mau de giai quyet Class Imbalance
    (cong thuc tu Pipeline_Joint_Learning.ipynb Cell 32).
    """
    valid_labels = [label for label in labels if label != -100]
    unique, counts = np.unique(valid_labels, return_counts=True)
    label_counts = dict(zip(unique, counts))
    weights = np.ones(num_classes)
    for i in range(num_classes):
        if i in label_counts:
            weights[i] = 1.0 / np.sqrt(label_counts[i])
        else:
            weights[i] = 1.0
    weights = weights / weights.sum() * num_classes
    return torch.FloatTensor(weights)


def main():
    print("[INFO] Bat dau qua trinh huan luyen va tao checkpoint 'best_model.pt'...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device su dung: {device}")

    # 1. Load encoders
    encoder_dir = os.path.join(root_dir, "models", "encoders")
    if not os.path.exists(encoder_dir):
        encoder_dir = os.path.join(root_dir, "streaming", "oneie_encoders")

    with open(os.path.join(encoder_dir, "trigger_encoder.pkl"), "rb") as f:
        trigger_encoder = pickle.load(f)
    with open(os.path.join(encoder_dir, "arg_encoder.pkl"), "rb") as f:
        arg_encoder = pickle.load(f)
    with open(os.path.join(encoder_dir, "event_encoder.pkl"), "rb") as f:
        event_encoder = pickle.load(f)

    num_triggers = len(trigger_encoder.classes_)
    num_args = len(arg_encoder.classes_)
    num_events = len(event_encoder.classes_)
    print(f"[INFO] So nhan: Trigger = {num_triggers}, Argument = {num_args}, Event Type = {num_events}")

    # 2. Load dataset
    csv_path = os.path.join(root_dir, "data", "results.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(root_dir, "results.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] File {csv_path} khong ton tai!")
        return

    print(f"[INFO] Doc du lieu tu {csv_path}...")
    df = pd.read_csv(csv_path)
    df["tokens"] = df["tokens"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df["bio_tags"] = df["bio_tags"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df["argument_tags"] = df["argument_tags"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Loc cac dong hop le
    df = df[
        (df["tokens"].str.len() == df["bio_tags"].str.len()) &
        (df["tokens"].str.len() == df["argument_tags"].str.len()) &
        (df["event_type"].notna())
    ].reset_index(drop=True)

    print(f"[INFO] Tong so mau hop le trong dataset: {len(df)}")

    # Chon subset dai dien can bang giua cac event types
    sample_size_per_class = 25
    train_indices = []
    for event_name in event_encoder.classes_:
        matched = df[df["event_type"] == event_name].index.tolist()
        train_indices.extend(matched[:sample_size_per_class])

    train_df = df.iloc[train_indices].reset_index(drop=True)
    print(f"[INFO] So mau huan luyen duoc can bang: {len(train_df)}")

    tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
    dataset = FastEventDataset(train_df, tokenizer, trigger_encoder, arg_encoder, event_encoder)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 3. Tinh toan class weights cho weighted loss
    all_t, all_a, all_e = [], [], []
    for batch in dataloader:
        all_t.extend(batch["trigger_labels"].view(-1).tolist())
        all_a.extend(batch["arg_labels"].view(-1).tolist())
        all_e.extend(batch["event_label"].tolist())

    trigger_weights = calculate_class_weights(all_t, num_triggers).to(device)
    arg_weights = calculate_class_weights(all_a, num_args).to(device)
    event_weights = calculate_class_weights(all_e, num_events).to(device)

    # 4. Khoi tao mo hinh
    model = JointModel("xlm-roberta-base", num_triggers, num_args, num_events)

    # Dong bang 10 lop dau de toi uu huan luyen tren CPU
    for name, param in model.xlm_roberta.named_parameters():
        if "layer.10" not in name and "layer.11" not in name and "pooler" not in name:
            param.requires_grad = False

    model.to(device)

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5)
    trigger_criterion = nn.CrossEntropyLoss(weight=trigger_weights, ignore_index=-100)
    arg_criterion = nn.CrossEntropyLoss(weight=arg_weights, ignore_index=-100)
    event_criterion = nn.CrossEntropyLoss(weight=event_weights)

    # 5. Huan luyen
    epochs = 2
    print(f"\n--- Bat dau huan luyen {epochs} epochs voi Weighted Loss ---")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            trigger_labels = batch["trigger_labels"].to(device)
            arg_labels = batch["arg_labels"].to(device)
            event_label = batch["event_label"].to(device)

            optimizer.zero_grad()
            trigger_logits, arg_logits, event_logits = model(input_ids, attention_mask)

            active_loss = attention_mask.view(-1) == 1
            t_loss = trigger_criterion(trigger_logits.view(-1, num_triggers)[active_loss], trigger_labels.view(-1)[active_loss])
            a_loss = arg_criterion(arg_logits.view(-1, num_args)[active_loss], arg_labels.view(-1)[active_loss])
            e_loss = event_criterion(event_logits, event_label)

            loss = 0.4 * t_loss + 0.3 * a_loss + 0.3 * e_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / max(batch_count, 1)
        print(f"[EPOCH {epoch+1}/{epochs}] Loss trung binh: {avg_loss:.4f}")

    duration = time.time() - start_time
    print(f"[INFO] Huan luyen hoan thanh trong {duration:.1f} giay!")

    # 6. Luu checkpoint vao models/checkpoints/best_model.pt
    save_dir = os.path.join(root_dir, "models", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pt")
    print(f"[INFO] Luu trong so mo hinh vao: {save_path}...")
    torch.save(model.state_dict(), save_path)
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"[SUCCESS] Da luu thanh cong {save_path} ({file_size_mb:.1f} MB)!")

    # 7. Test thu inference
    eval_model = JointModel("xlm-roberta-base", num_triggers, num_args, num_events)
    eval_model.load_state_dict(torch.load(save_path, map_location=device))
    eval_model.eval()
    print("[SUCCESS] Nap state_dict vao JointModel thanh cong va san sang su dung!")


if __name__ == "__main__":
    main()
