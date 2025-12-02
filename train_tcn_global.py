# train_tcn_global.py — FINAL BULLETPROOF GLOBAL TCN (DEC 2025)
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# === YOUR EXACT 75 FEATURES ===
FEATURE_NAMES = [
    "ret_24h", "price_z_120h", "ret_1h", "bb_position_1h", "ret_4h", "adx_1h", "adx_4h",
    "bb_position_1h.2", "bb_position_1h.1", "ema10_change_1d", "ema10_change_1h", "ema10_change_4h",
    "ema10_dev_1d", "ema10_dev_1h", "ema10_dev_4h", "macd_1d", "macd_1h", "macd_4h",
    "macd_change_1d", "macd_change_1h", "macd_change_4h", "pattern_bearish_engulfing_1d",
    "pattern_bearish_engulfing_1h", "pattern_bearish_engulfing_4h", "pattern_bullish_engulfing_1d",
    "pattern_bullish_engulfing_1h", "pattern_bullish_engulfing_4h", "pattern_evening_star_1d",
    "pattern_evening_star_1h", "pattern_evening_star_4h", "pattern_hammer_1d", "pattern_hammer_1h",
    "pattern_hammer_4h", "pattern_marubozu_bear_1d", "pattern_marubozu_bear_1h",
    "pattern_marubozu_bear_4h", "pattern_marubozu_bull_1d", "pattern_marubozu_bull_1h",
    "pattern_marubozu_bull_4h", "pattern_morning_star_1d", "pattern_morning_star_1h",
    "pattern_morning_star_4h", "pattern_shooting_star_1d", "pattern_shooting_star_1h",
    "pattern_shooting_star_4h", "price_above_ema10_1d", "price_above_ema10_1h", "price_above_ema10_4h",
    "price_z_120h.2", "price_z_120h.1", "ret_1h.2", "ret_1h.1", "ret_24h.2", "ret_24h.1",
    "ret_4h.2", "ret_4h.1", "rsi_1h", "rsi_4h", "rsi_change_1h", "signal_1d", "signal_1h",
    "signal_4h", "sp500_above_20d", "stoch_d_1d", "stoch_d_1h", "stoch_d_4h", "stoch_k_1d",
    "stoch_k_1h", "stoch_k_4h", "td9_1d", "td9_1h", "td9_4h", "zig_1d", "zig_1h", "zig_4h"
]

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.25):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = nn.Conv1d(out_channels, out_channels, 1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = nn.Conv1d(out_channels, out_channels, 1)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        out = out[:, :, :x.size(2)]  # Chomp the padding
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNGlobal(nn.Module):
    def __init__(self, n_features=len(FEATURE_NAMES), channels=[128]*8, kernel_size=3, dropout=0.25):
        super().__init__()
        layers = []
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            in_ch = n_features if i == 0 else channels[i-1]
            layers.append(TemporalBlock(in_ch, ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels[-1], 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, S, F) → (B, F, S)
        x = self.network(x)
        x = x[:, :, -1]  # last timestep
        x = self.dropout(x)
        return self.classifier(x).squeeze(-1)

# === TRAINING ===
if __name__ == "__main__":
    logger.info("Loading global dataset...")
    df = pd.read_csv("/app/data/historical_data_no_price.csv")
    if "decision" not in df.columns:
        raise ValueError("Run add_decision_column.py first!")

    y = (df["decision"] == 0).astype(np.float32).values  # 1=Buy
    X_raw = df[FEATURE_NAMES].fillna(0).values.astype(np.float32)

    seq_len = 120
    logger.info("Creating sequences...")
    X_seq, y_seq = [], []
    for i in range(len(X_raw) - seq_len - 7):
        X_seq.append(X_raw[i:i + seq_len])
        y_seq.append(y[i + seq_len - 1])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    split = int(len(X_seq) * 0.85)
    X_train = torch.from_numpy(X_seq[:split])
    y_train = torch.from_numpy(y_seq[:split])
    X_val = torch.from_numpy(X_seq[split:])
    y_val = torch.from_numpy(y_seq[split:])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCNGlobal().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-5)

    best_auc = 0.0
    for epoch in range(1, 121):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = torch.sigmoid(model(X_val.to(device))).cpu().numpy()
            auc = roc_auc_score(y_val.numpy(), val_pred)
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), "/app/models/tcn_global_best.pt")
                logger.info(f"EPOCH {epoch:3d} | NEW BEST AUC: {auc:.5f} → SAVED")

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:3d} | Val AUC: {auc:.5f} | Best: {best_auc:.5f}")

    logger.info(f"FINAL GLOBAL TCN → BEST AUC: {best_auc:.5f}")
    logger.info("Model saved: /app/models/tcn_global_best.pt")