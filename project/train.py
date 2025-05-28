import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, get_scheduler
from datasets import load_dataset
from model import CrossAttentionModel
from utils import set_seed, STSDataset, tokenize_function
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import numpy as np
import matplotlib.pyplot as plt

class CustomDataCollatorWithPadding:
    def __init__(self, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding_collator = DataCollatorWithPadding(tokenizer, padding='max_length', max_length=max_length)
    
    def __call__(self, examples):
        features1 = [{'input_ids': ex['input_ids1'], 'attention_mask': ex['attention_mask1']} for ex in examples]
        features2 = [{'input_ids': ex['input_ids2'], 'attention_mask': ex['attention_mask2']} for ex in examples]
        labels = [ex['labels'] for ex in examples]
        batch1 = self.padding_collator(features1)
        batch2 = self.padding_collator(features2)
        batch = {
            'input_ids1': batch1['input_ids'].to(dtype=torch.long),
            'attention_mask1': batch1['attention_mask'].to(dtype=torch.long),
            'input_ids2': batch2['input_ids'].to(dtype=torch.long),
            'attention_mask2': batch2['attention_mask'].to(dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }
        return batch

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar y preprocesar dataset
dataset = load_dataset("mteb/stsbenchmark-sts")

# Normalizar puntajes a [0, 5]
def normalize_scores(example):
    example['labels'] = float(example['score'])  # Asegurar float, ya en [0, 5]
    return example

dataset = dataset.map(normalize_scores)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")

# Ajustar tokenize_function para tokenizar sentence1 y sentence2 por separado
def tokenize_function(example, tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    encodings1 = tokenizer(
        example['sentence1'],
        padding='max_length',
        truncation=True,
        max_length=64,
        return_tensors=None
    )
    encodings2 = tokenizer(
        example['sentence2'],
        padding='max_length',
        truncation=True,
        max_length=64,
        return_tensors=None
    )
    return {
        'input_ids1': encodings1['input_ids'],
        'attention_mask1': encodings1['attention_mask'],
        'input_ids2': encodings2['input_ids'],
        'attention_mask2': encodings2['attention_mask'],
        'labels': example['labels']
    }

tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=False)

# Debug: Verificar estructura del dataset
print("Dataset keys:", dataset.keys())
print("Train dataset keys:", dataset["train"].column_names)
print("Sample train data:", dataset["train"][0])
print("Score range in train:", min(dataset["train"]["labels"]), "to", max(dataset["train"]["labels"]))

# Crear datasets y dataloaders
train_dataset = STSDataset(tokenized_datasets["train"])
val_dataset = STSDataset(tokenized_datasets["validation"])
data_collator = CustomDataCollatorWithPadding(tokenizer=tokenizer, max_length=64)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=data_collator)

# Crear directorio de checkpoints
os.makedirs("checkpoints", exist_ok=True)

# Inicializar modelo con menos capas congeladas
model = CrossAttentionModel(model_name="sentence-transformers/all-MiniLM-L12-v2", freeze_layers=6).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)  # Aumentar lr
num_epochs = 5
num_training_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * num_training_steps)
scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
)

# Simplificar función de pérdida
class ScaledMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
    
    def forward(self, outputs, labels):
        return self.mse(outputs, labels), outputs  # Usar clamp en model.forward

loss_fn = ScaledMSELoss()
best_pearson = -float("inf")
best_model_path = "checkpoints/best_model_cross.pt"

# Listas para métricas
train_losses = []
val_losses = []
pearson_scores = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        input_ids1 = batch["input_ids1"].to(device)
        attention_mask1 = batch["attention_mask1"].to(device)
        input_ids2 = batch["input_ids2"].to(device)
        attention_mask2 = batch["attention_mask2"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
        loss, scaled_outputs = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Evaluar en validación
    model.eval()
    preds, labels = [], []
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids1 = batch["input_ids1"].to(device)
            attention_mask1 = batch["attention_mask1"].to(device)
            input_ids2 = batch["input_ids2"].to(device)
            attention_mask2 = batch["attention_mask2"].to(device)
            lbls = batch["labels"].to(device)
            outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            loss, scaled_outputs = loss_fn(outputs, lbls)
            total_val_loss += loss.item()
            preds.extend(scaled_outputs.cpu().numpy())
            labels.extend(lbls.cpu().numpy())
    
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    try:
        pearson_corr, _ = pearsonr(preds, labels)
    except ValueError:
        pearson_corr = float('nan')
    pearson_scores.append(pearson_corr)
    mae = mean_absolute_error(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}")
    print(f"  Pearson: {pearson_corr:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    if not np.isnan(pearson_corr) and pearson_corr > best_pearson:
        best_pearson = pearson_corr
        torch.save({'model_state_dict': model.state_dict()}, best_model_path)  # Sin weights_only
        print(f"  ✓ Mejor modelo guardado (Pearson: {best_pearson:.4f})")

    # Gráfico de distribución
    plt.figure(figsize=(8, 6))
    plt.hist(preds, bins=20, range=(0, 5), alpha=0.7, label='Predicciones', color='blue')
    plt.hist(labels, bins=20, range=(0, 5), alpha=0.7, label='Etiquetas', color='orange')
    plt.legend()
    plt.xlabel('Puntaje')
    plt.ylabel('Frecuencia')
    plt.title(f'Distribución de Predicciones vs Etiquetas (Época {epoch+1})')
    plt.savefig(f'checkpoints/pred_distribution_epoch_{epoch+1}.png')
    plt.close()

print(f"Entrenamiento finalizado. Mejor modelo guardado en {best_model_path}")