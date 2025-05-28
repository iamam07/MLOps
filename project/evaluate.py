import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from model import CrossAttentionModel
from utils import tokenize_function, STSDataset
from scipy.stats import pearsonr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar y tokenizar dataset de test
dataset = load_dataset("mteb/stsbenchmark-sts")
tokenized_test = dataset["test"].map(tokenize_function, batched=True)
test_dataset = STSDataset(tokenized_test)
data_collator = DataCollatorWithPadding(tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2"), max_length=128)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=data_collator)

# Cargar modelo entrenado
model = CrossAttentionModel().to(device)
model.load_state_dict(torch.load("checkpoints/best_model_cross.pt")['model_state_dict'])
model.eval()

# Evaluar
preds, labels = [], []
total_loss = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids1 = batch["input_ids1"].to(device)
        attention_mask1 = batch["attention_mask1"].to(device)
        input_ids2 = batch["input_ids2"].to(device)
        attention_mask2 = batch["attention_mask2"].to(device)
        lbls = batch["labels"].to(device)
        outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
        loss = torch.nn.MSELoss()(outputs, lbls)
        total_loss += loss.item()
        preds.extend(outputs.cpu().numpy())
        labels.extend(lbls.cpu().numpy())

avg_loss = total_loss / len(test_loader)
correlation, _ = pearsonr(preds, labels)
print(f"Test Loss: {avg_loss:.4f}, Pearson: {correlation:.4f}")