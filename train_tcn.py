# train_tcn.py — 终极 TCN 训练脚本（已完美适配你的环境）
import argparse
import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp = nn.utils.weight_norm(nn.Conv1d(out_channels, out_channels, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        if self.downsample:
            nn.init.xavier_uniform_(self.downsample.weight)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]]
        gate = torch.sigmoid(self.chomp(out))
        out = self.relu(out) * gate
        out = self.norm(self.dropout(out))
        if self.downsample:
            residual = self.downsample(residual)
        return out + residual

class TCN(nn.Module):
    def __init__(self, n_features, channels=[64]*8, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(channels)):
            dilation = 2 ** i
            in_ch = n_features if i == 0 else channels[i-1]
            out_ch = channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels[-1], 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.network(x)
        y = y[:, :, -1]
        return torch.sigmoid(self.classifier(y)).squeeze(-1)

class TradingDataset(Dataset):
    def __init__(self, df, seq_len=60):
        self.seq_len = seq_len
        self.features = [c for c in df.columns if c not in ["timestamp", "ticker", "decision", "price"]]
        self.X = df[self.features].values.astype(np.float32)
        # Directly learn from the ``decision`` column so labeling pipelines can
        # control the target encoding (e.g., 0=Buy, 1=Sell from
        # ``add_decision_column.py``).
        self.y = df["decision"].astype(np.float32).values
        self.valid_idx = np.arange(seq_len, len(df))

    def __len__(self): return len(self.valid_idx)
    def __getitem__(self, idx):
        i = self.valid_idx[idx]
        return torch.from_numpy(self.X[i-self.seq_len:i]), torch.tensor(self.y[i])

def train_tcn(ticker="TQQQ", epochs=60):
    data_path = Path("/app/data/historical_data_no_price.csv")
    if not data_path.exists():
        logger.error("数据文件不存在！请先运行数据收集")
        return

    df = pd.read_csv(data_path)
    df = df[df["ticker"] == ticker].copy()

    # 只用有信号的行训练
    df = df[df["decision"].isin([0, 1])].copy()
    if len(df) < 100:
        logger.error(f"{ticker} 有信号样本太少")
        return
    logger.info(f"训练样本: {len(df)} 条 | Buy(1): {sum(df['decision']==1)} | Sell(0): {sum(df['decision']==0)}")

    split = int(len(df) * 0.85)
    train_df, val_df = df.iloc[:split], df.iloc[split:]

    train_set = TradingDataset(train_df)
    val_set = TradingDataset(val_df)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = len(train_set.features)
    logger.info(f"检测到 {n_features} 个特征，自动构建 TCN...")
    model = TCN(n_features=n_features).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # 验证
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X, y in val_loader:
                pred = model(X.to(device)).cpu().numpy()
                preds.extend(pred)
                trues.extend(y.numpy())
        auc = roc_auc_score(trues, preds)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f"/app/models/tcn_best_{ticker.lower()}.pt")
            logger.info(f"NEW BEST → AUC: {auc:.4f} | 模型已保存")

        scheduler.step()
        if epoch % 10 == 0 or epoch == epochs:
            logger.info(f"Epoch {epoch}/{epochs} | AUC: {auc:.4f} | Best: {best_auc:.4f}")

    logger.info(f"TCN 训练完成！最佳 AUC: {best_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="TQQQ")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    train_tcn(args.ticker.upper(), args.epochs)

