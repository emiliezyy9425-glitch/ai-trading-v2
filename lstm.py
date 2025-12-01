# /app/models/lstm.py
import torch
import torch.nn as nn
import math


class AttentiveBiLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.35,
        seq_len: int = 60,
        bidirectional: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.seq_len = seq_len

        # === INPUT PROJECTION + SCALING (critical!) ===
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.layer_norm_in = nn.LayerNorm(hidden_size)

        # === BIDIRECTIONAL LSTM WITH RESIDUAL + LAYER NORM ===
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Layer norms after LSTM
        lstm_out_size = hidden_size * 2 if bidirectional else hidden_size
        self.layer_norm_lstm = nn.LayerNorm(lstm_out_size)

        # === ATTENTION (query from last hidden state) ===
        self.attention_query = nn.Linear(lstm_out_size, lstm_out_size)
        self.attention_key = nn.Linear(lstm_out_size, lstm_out_size)
        self.attention_val = lstm_out_size  # for softmax

        # === FINAL CLASSIFIER ===
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, lstm_out_size // 2),
            nn.GELU(),
            nn.LayerNorm(lstm_out_size // 2),
            nn.Dropout(0.2),
            nn.Linear(lstm_out_size // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size = x.size(0)

        # Input projection + scaling
        x = self.input_proj(x) * math.sqrt(self.hidden_size)
        x = self.layer_norm_in(x)

        # LSTM
        lstm_out, (h_n, _) = self.lstm(x)  # lstm_out: (B, S, D*H)
        lstm_out = self.layer_norm_lstm(lstm_out)

        # === ATTENTION ===
        # Use last hidden state as query
        if self.bidirectional:
            query = torch.cat((h_n[-2], h_n[-1]), dim=-1)  # (B, D*H)
        else:
            query = h_n[-1]  # (B, H)
        query = self.attention_query(query).unsqueeze(1)  # (B, 1, D*H)

        keys = self.attention_key(lstm_out)  # (B, S, D*H)
        attn_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)  # (B, S)
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (B, S, 1)

        context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, D*H)

        # Final prediction
        out = self.dropout(context)
        logits = self.classifier(out)
        return logits.squeeze(-1)  # (batch_size,)
