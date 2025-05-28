import torch.nn as nn
from transformers import AutoModel

class CrossEncoderRegressionModel(nn.Module):
    def __init__(self, model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Asegura salida en [0, 1]
        )

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.last_hidden_state[:, 0, :]
        return self.regressor(cls_embedding).squeeze(-1)