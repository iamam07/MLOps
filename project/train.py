import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_scheduler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import numpy as np
import os

from model import CrossEncoderRegressionModel
from utils import STSDataset, set_seed

# Config
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
BATCH_SIZE = 8
LR = 2e-5 #1e-5  
EPOCHS = 15

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
dataset = load_dataset("mteb/stsbenchmark-sts")
dataset = dataset.map(lambda x: {"score": [s / 5.0 for s in x["score"]]}, batched=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

tokenized = dataset.map(tokenize_function, batched=True)

train_dataset = STSDataset(tokenized["train"])
val_dataset = STSDataset(tokenized["validation"])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
model = CrossEncoderRegressionModel(model_name=MODEL_NAME).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * len(train_loader) * EPOCHS),
    num_training_steps=len(train_loader) * EPOCHS,
)

best_pearson = -1.0
OUTPUT_MODEL_PATH = os.getenv("OUTPUT_MODEL_PATH", "checkpoints/best_model.pt")
os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        preds = model(input_ids, attention_mask)
        loss = torch.nn.MSELoss()(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids, attention_mask)
            preds.extend(out.cpu().numpy())
            golds.extend(labels.cpu().numpy())

    pearson, _ = pearsonr(preds, golds)
    mae = mean_absolute_error(golds, preds)
    rmse = np.sqrt(mean_squared_error(golds, preds))

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Pearson: {pearson:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")

    if pearson > best_pearson:
        best_pearson = pearson
        torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
        print("  ✓ Mejor modelo guardado.")

print("Entrenamiento finalizado.")
