"""Retrain pipeline entrypoint.

Currently supports the Transformer model used by the live agent. It loads
hyperparameters from a JSON config file and trains the classifier on the
cleaned historical dataset with the leak-free labeling scheme.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models.transformer import TransformerModel
from project_paths import resolve_data_path
from self_learn import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("/app/models")


def train_transformer(params: Dict[str, Any]) -> None:
    logger.info("=== STARTING FINAL GLOBAL TRANSFORMER TRAINING (PERFECT LABEL + NO LEAK) ===")

    # 1. LOAD YOUR GOLDEN DATA WITH THE PERFECT LABEL
    data_path = resolve_data_path("historical_data_no_price.csv")
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        return

    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    logger.info(f"Loaded {len(df):,} rows from {data_path.name}")

    # 2. FINAL SAFETY: REMOVE ANY RAW PRICE (should already be gone)
    price_cols = ["open", "high", "low", "close", "price", "adj_close"]
    dropped = [c for c in price_cols if c in df.columns]
    if dropped:
        logger.warning(f"Dropping raw price columns: {dropped}")
        df.drop(columns=dropped, inplace=True)

    # 3. USE YOUR PERFECT LABEL: decision (0=Buy, 1=Sell → from 8h forward return)
    if "decision" not in df.columns:
        logger.error("decision column missing! Run add_decision_column.py first!")
        return

    # Convert: 0=Buy → 1.0, 1=Sell → 0.0 (standard binary classification)
    y = (df["decision"] == 0).astype(np.float32).values  # 1.0 = Buy, 0.0 = Sell
    X = df[FEATURE_NAMES].fillna(0).values.astype(np.float32)

    seq_len = params["time_steps"]

    # 4. NO-LEAK SEQUENCE CREATION (safe for 8h forward label)
    def create_sequences_safe(X: np.ndarray, y: np.ndarray, seq_len: int):
        max_start = len(X) - seq_len - 8 + 1  # +8 because label looks 8h ahead
        if max_start <= 0:
            return np.array([]), np.array([])
        seqs = np.array([X[i : i + seq_len] for i in range(max_start)])
        labels = y[seq_len + 7 : seq_len + 7 + len(seqs)]  # label at end of sequence + 8h
        return seqs, labels

    X_seq, y_seq = create_sequences_safe(X, y, seq_len)
    if len(X_seq) < 5000:
        logger.error("Not enough clean sequences! Check data.")
        return

    logger.info(f"Created {len(X_seq):,} clean sequences (8h forward label safe)")

    # 5. GLOBAL TRAIN/VAL SPLIT
    split_idx = int(len(X_seq) * 0.85)
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

    # 6. GLOBAL SCALER
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])

    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)

    # 7. DATASETS
    train_dataset = TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val_scaled), torch.from_numpy(y_val))

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)

    # 8. MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(
        input_size=len(FEATURE_NAMES),
        d_model=params["d_model"],
        nhead=params["nhead"],
        num_layers=params["num_layers"],
        dim_feedforward=params["dim_feedforward"],
        dropout=params["dropout"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.1))  # slight help for minority class
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=1e-5)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_val_auc = 0.0
    patience = 15
    patience_counter = 0

    for epoch in range(params["epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        all_probs: list[float] = []
        all_labels: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(yb.numpy())

        val_auc = roc_auc_score(all_labels, all_probs)
        logger.info(f"Epoch {epoch + 1:3d} | Val AUC: {val_auc:.5f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_DIR / "updated_transformer.pt")
            joblib.dump(scaler, MODEL_DIR / "transformer_scaler.joblib")
            logger.info("  → NEW BEST MODEL SAVED")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping")
                break

    logger.info(f"Training complete! Best Val AUC: {best_val_auc:.5f}")
    logger.info("FINAL MODEL: updated_transformer.pt")
    logger.info("SCALER:      transformer_scaler.joblib")
    logger.info("=== TRANSFORMER IS NOW NUCLEAR ===")


def load_params(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain supported ML models")
    parser.add_argument("--config", type=str, default="transformer_config.json", help="Path to the config JSON")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return

    params = load_params(config_path)
    if "transformer" in config_path.stem:
        train_transformer(params)
    else:
        logger.info(f"No retraining routine implemented for {config_path.name}; skipping.")


if __name__ == "__main__":
    main()
