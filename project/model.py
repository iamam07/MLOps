import torch
import torch.nn as nn
from transformers import AutoModel

class CrossAttentionModel(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L12-v2", freeze_layers=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        for i, param in enumerate(self.bert.encoder.layer):
            if i < (len(self.bert.encoder.layer) - freeze_layers):
                for p in param.parameters():
                    p.requires_grad = False
        self.cross_attention = nn.MultiheadAttention(embed_dim=384, num_heads=8)
        self.regressor = nn.Sequential(
            nn.Linear(384 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
            # Removed Sigmoid - direct output for 0-5 range
        )
        # Initialize regressor weights
        for layer in self.regressor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        output1 = self.bert(input_ids1, attention_mask=attention_mask1).last_hidden_state
        output2 = self.bert(input_ids2, attention_mask=attention_mask2).last_hidden_state

        output1 = output1.permute(1, 0, 2)
        output2 = output2.permute(1, 0, 2)

        key_padding_mask = ~attention_mask2.bool()

        attn_output, _ = self.cross_attention(
            query=output1,
            key=output2,
            value=output2,
            key_padding_mask=key_padding_mask
        )

        attn_output = attn_output.permute(1, 0, 2)

        mask1 = attention_mask1.unsqueeze(-1).float()
        mask2 = attention_mask2.unsqueeze(-1).float()

        pooled_orig = torch.sum(output1.permute(1, 0, 2) * mask1, dim=1) / torch.clamp(mask1.sum(dim=1), min=1e-9)
        pooled_attended = torch.sum(attn_output * mask1, dim=1) / torch.clamp(mask1.sum(dim=1), min=1e-9)
        pooled_output2 = torch.sum(output2.permute(1, 0, 2) * mask2, dim=1) / torch.clamp(mask2.sum(dim=1), min=1e-9)

        combined = torch.cat([pooled_orig, pooled_attended, pooled_output2], dim=1)
        output = self.regressor(combined).squeeze()

        # Clamp output to valid range [0, 5]
        return torch.clamp(output, 0, 5)