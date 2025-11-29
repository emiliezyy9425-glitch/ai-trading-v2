#!/usr/bin/env python3
"""
self_learn.py
Central feature registry and self-learning utilities.
Synchronized with the live trading FEATURE_NAMES schema (patterns + macro inputs).
"""
import os
import argparse
import json
from datetime import datetime, timezone
from typing import List, Dict, Any

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

# ================================
# FEATURE NAMES — AUTO-GENERATED FROM CSV
# ================================

# Base feature names (without suffix) used by derived helpers
_BASE_FEATURES: List[str] = [
    "rsi",
    "macd",
    "signal",
    "ema10",
    "ema10_dev",
    "rsi_change",
    "macd_change",
    "ema10_change",
    "price_above_ema10",
    "bb_upper",
    "bb_lower",
    "bb_mid",
    "volume",
    "td9",
    "zig",
    "atr",
    "adx",
    "obv",
    "stoch_k",
    "stoch_d",
]

# Explicit per-timeframe feature schema (order matters for model compatibility)
FEATURE_NAMES_1H: List[str] = [
    "rsi_1h",
    "macd_1h",
    "signal_1h",
    "ema10_1h",
    "ema10_dev_1h",
    "rsi_change_1h",
    "macd_change_1h",
    "ema10_change_1h",
    "price_above_ema10_1h",
    "bb_upper_1h",
    "bb_lower_1h",
    "bb_mid_1h",
    "volume_1h",
    "td9_1h",
    "zig_1h",
    "atr_1h",
    "adx_1h",
    "obv_1h",
    "stoch_k_1h",
    "stoch_d_1h",
    # Golden price derivatives (retain order with 1h context)
    "ret_1h",
    "ret_4h",
    "ret_24h",
    "price_z_120h",
    "bb_position_1h",
]

FEATURE_NAMES_4H: List[str] = [
    "rsi_4h",
    "macd_4h",
    "signal_4h",
    "ema10_4h",
    "ema10_dev_4h",
    "macd_change_4h",
    "ema10_change_4h",
    "price_above_ema10_4h",
    "bb_upper_4h",
    "bb_lower_4h",
    "bb_mid_4h",
    "volume_4h",
    "td9_4h",
    "zig_4h",
    "atr_4h",
    "adx_4h",
    "obv_4h",
    "stoch_k_4h",
    "stoch_d_4h",
    # Golden price derivatives at 4h cadence
    "ret_4h",
    "ret_4h_4h",
    "ret_24h_4h",
    "price_z_120h_4h",
    "bb_position_4h",
]

FEATURE_NAMES_1D: List[str] = [
    "macd_1d",
    "signal_1d",
    "ema10_1d",
    "ema10_dev_1d",
    "macd_change_1d",
    "ema10_change_1d",
    "price_above_ema10_1d",
    "bb_upper_1d",
    "bb_lower_1d",
    "bb_mid_1d",
    "volume_1d",
    "td9_1d",
    "zig_1d",
    "atr_1d",
    "obv_1d",
    "stoch_k_1d",
    "stoch_d_1d",
    # Golden price derivatives at 1d cadence
    "ret_1d",
    "ret_4h_1d",
    "ret_24h_1d",
    "price_z_120h_1d",
    "bb_position_1d",
]

_PATTERN_BASES: List[str] = [
    "pattern_bullish_engulfing",
    "pattern_bearish_engulfing",
    "pattern_hammer",
    "pattern_shooting_star",
    "pattern_marubozu_bull",
    "pattern_marubozu_bear",
    "pattern_morning_star",
    "pattern_evening_star",
]

_PATTERN_TIMEFRAMES: List[str] = ["1h", "4h", "1d"]

CANDLESTICK_FEATURES: List[str] = [
    f"{base}_{tf}"
    for tf in _PATTERN_TIMEFRAMES
    for base in _PATTERN_BASES
]

ADDITIONAL_FEATURES: List[str] = ["sp500_above_20d"]

# Full list for RF, PPO, ensemble
FEATURE_NAMES: List[str] = (
    FEATURE_NAMES_1H
    + FEATURE_NAMES_4H
    + FEATURE_NAMES_1D
    + ADDITIONAL_FEATURES
    + CANDLESTICK_FEATURES
)

# ================================
# PATHS
# ================================
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/app")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CURATED_DIR = os.path.join(DATA_DIR, "lake", "curated")
TRADE_LOG_DEFAULT = os.getenv("TRADE_LOG_PATH", os.path.join(DATA_DIR, "trade_log.csv"))
HIST_FEATURES = os.path.join(DATA_DIR, "historical_data_no_price.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CURATED_DIR, exist_ok=True)

# ================================
# DATA LOADING
# ================================
def load_inputs(trade_log_path: str, generate_missing: bool = True):
    """Load trade log and curated features."""
    # Placeholder — your original logic here
    # Return (trades_df, features_df)
    return pd.DataFrame(), pd.DataFrame()

def label_trades(trades, feats, horizon_days, threshold):
    """Label trades as good/bad."""
    # Placeholder
    return pd.DataFrame()

# ================================
# RANDOM FOREST TRAINING
# ================================
def train_rf(labeled: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
    """Train RF on labeled data."""
    X = labeled[FEATURE_NAMES].fillna(0).values
    y = labeled["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = os.path.join(MODEL_DIR, "rf_latest.joblib")
    archive_path = os.path.join(MODEL_DIR, f"rf_{stamp}.joblib")
    joblib.dump(clf, model_path)
    joblib.dump(clf, archive_path)

    meta = {
        "created_at_utc": stamp,
        "samples": int(len(labeled)),
        "positives": int(labeled["label"].sum()),
        "negatives": int(len(labeled) - labeled["label"].sum()),
        "feature_names": FEATURE_NAMES,
        "test_report": report
    }
    with open(os.path.join(MODEL_DIR, "rf_latest.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {"model_path": model_path, "archive_path": archive_path, "report": report}

# ================================
# CLI
# ================================
def main():
    ap = argparse.ArgumentParser(description="Self-learn from trade_log to update RF model.")
    ap.add_argument("--horizon-days", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--no-generate-historical", action="store_false", dest="generate_historical")
    ap.add_argument("--trade-log", default=TRADE_LOG_DEFAULT)
    ap.set_defaults(generate_historical=True)
    args = ap.parse_args()

    trades, feats = load_inputs(args.trade_log, generate_missing=args.generate_historical)
    if trades.empty:
        print("No trades to learn from.")
        return
    labeled = label_trades(trades, feats, args.horizon_days, args.threshold)
    if labeled.empty or labeled['label'].nunique() < 2:
        print("Not enough labeled samples.")
        return
    result = train_rf(labeled, test_size=args.test_size)
    print(f"Model saved to: {result['model_path']}")
    print(json.dumps(result["report"], indent=2))

if __name__ == "__main__":
    main()
