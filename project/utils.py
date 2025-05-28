import random
import numpy as np
import torch
from transformers import AutoTokenizer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

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

class STSDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        print("Available columns in dataset:", data.column_names)
        print("Dataset length:", len(data))
        self.input_ids1 = data["input_ids1"]
        self.attention_mask1 = data["attention_mask1"]
        self.input_ids2 = data["input_ids2"]
        self.attention_mask2 = data["attention_mask2"]
        if "labels" in data.column_names:
            self.scores = data["labels"]
            print(f"Found 'labels' column with {len(self.scores)} values")
        else:
            print("Warning: 'labels' column not found, using zeros")
            self.scores = [0] * len(data)
        print("Score statistics:")
        print(f"  Min: {min(self.scores):.2f}")
        print(f"  Max: {max(self.scores):.2f}")
        print(f"  Mean: {np.mean(self.scores):.2f}")

    def __len__(self):
        return len(self.input_ids1)

    def __getitem__(self, idx):
        return {
            'input_ids1': self.input_ids1[idx],
            'attention_mask1': self.attention_mask1[idx],
            'input_ids2': self.input_ids2[idx],
            'attention_mask2': self.attention_mask2[idx],
            'labels': self.scores[idx]
        }