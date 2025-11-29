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

# Base feature names (without suffix)
_BASE_FEATURES: List[str] = [
    # Core Technicals
    "rsi", "macd", "signal", "ema10", "ema10_dev",
    "rsi_change", "macd_change", "ema10_change",
    "price_above_ema10", "bb_upper", "bb_lower", "bb_mid",
    "volume", "tds_trend", "tds_signal", "high_vol", "vol_spike",
    "td9", "zig", "vol_cat",

    # Fibonacci
    "fib_level1", "fib_level2", "fib_level3", "fib_level4", "fib_level5", "fib_level6",
    "fib_time_count", "fib_zone_delta",

    # Volatility & Trend
    "atr", "adx", "obv", "stoch_k", "stoch_d",
]

# Multi-timeframe versions
def _make_suffixed(names: List[str], suffix: str) -> List[str]:
    return [f"{name}_{suffix}" for name in names]

FEATURE_NAMES_1H: List[str] = _make_suffixed(_BASE_FEATURES, "1h")
FEATURE_NAMES_4H: List[str] = _make_suffixed(_BASE_FEATURES, "4h")
FEATURE_NAMES_1D: List[str] = _make_suffixed(_BASE_FEATURES, "1d")

# self_learn.py → FINAL FEATURE_NAMES (candlestick part only)
_PATTERN_BASES: List[str] = [
    "pattern_bullish_engulfing",
    "pattern_bearish_engulfing",
    "pattern_marubozu_bull",
    "pattern_marubozu_bear",
    # Newly added to align live features with trained models/scalers
    "pattern_hammer",
    "pattern_shooting_star",
    "pattern_morning_star",
    "pattern_evening_star",
]

_PATTERN_TIMEFRAMES: List[str] = ["1h", "4h", "1d"]

CANDLESTICK_FEATURES: List[str] = [
    f"{base}_{tf}"
    for base in _PATTERN_BASES
    for tf in _PATTERN_TIMEFRAMES
]

ADDITIONAL_FEATURES: List[str] = [
    "iv",
    "delta",
    "sp500_above_20d",
    "level_weight",
    "iv_atm",
    "delta_atm_call",
    "delta_atm_put",
    "spx_above_20d",
    "vix",
    "iv_rank_proxy",
]

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
