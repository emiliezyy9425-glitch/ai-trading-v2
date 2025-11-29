# /app/models/lstm.py
import torch
import torch.nn as nn

class AttentiveBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3, seq_len=64):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )

        # Attention over time
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()  # ← ADD THIS
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (B, S, 2*H)

        # Attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (B, S, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, 2*H)

        out = self.dropout(context)
        out = self.fc(out)
        return self.sigmoid(out)  # ← ADD THIS