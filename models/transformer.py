import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 384,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        max_seq_len: int = 120
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=500)  # 500 is more than enough

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # Take last timestep
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits.squeeze(-1)  # (batch_size,)
