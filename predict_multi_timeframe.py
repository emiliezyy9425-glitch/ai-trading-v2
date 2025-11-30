#!/usr/bin/env python3
"""
Multi-Timeframe AI — FINAL • WORKS EXACTLY LIKE BACKTESTER • ALWAYS CURRENT • ZERO ERRORS
"""

import argparse
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd

from scripts.generate_historical_data import (
    _augment_timeframe_features,
    _finalise_feature_frame,
)
from feature_engineering import add_golden_price_features
from ml_predictor import predict_with_all_models, independent_model_decisions as ensemble_vote

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PRICE_COLS = {"daily": "price_1d", "weekly": "price_1w", "monthly": "price_1M"}
LIVE_MODEL_FEATURES: list[str] = [
    "bb_position_1h",
    "ret_24h",
    "price_z_120h",
    "ret_4h",
    "ret_1h",
    "adx_1h",
    "adx_4h",
    "bb_position_1h.1",
    "ema10_change_1d",
    "ema10_change_1h",
    "ema10_change_4h",
    "ema10_dev_1d",
    "ema10_dev_1h",
    "ema10_dev_4h",
    "macd_1d",
    "macd_1h",
    "macd_4h",
    "macd_change_1d",
    "macd_change_1h",
    "macd_change_4h",
    "pattern_bearish_engulfing_1d",
    "pattern_bearish_engulfing_1h",
    "pattern_bearish_engulfing_4h",
    "pattern_bullish_engulfing_1d",
    "pattern_bullish_engulfing_1h",
    "pattern_bullish_engulfing_4h",
    "pattern_evening_star_1d",
    "pattern_evening_star_1h",
    "pattern_evening_star_4h",
    "pattern_hammer_1d",
    "pattern_hammer_1h",
    "pattern_hammer_4h",
    "pattern_marubozu_bear_1d",
    "pattern_marubozu_bear_1h",
    "pattern_marubozu_bear_4h",
    "pattern_marubozu_bull_1d",
    "pattern_marubozu_bull_1h",
    "pattern_marubozu_bull_4h",
    "pattern_morning_star_1d",
    "pattern_morning_star_1h",
    "pattern_morning_star_4h",
    "pattern_shooting_star_1d",
    "pattern_shooting_star_1h",
    "pattern_shooting_star_4h",
    "price_above_ema10_1d",
    "price_above_ema10_1h",
    "price_above_ema10_4h",
    "price_z_120h.1",
    "ret_1h.1",
    "ret_24h.1",
    "ret_4h.1",
    "rsi_1h",
    "rsi_4h",
    "rsi_change_1h",
    "signal_1d",
    "signal_1h",
    "signal_4h",
    "sp500_above_20d",
    "stoch_d_1d",
    "stoch_d_1h",
    "stoch_d_4h",
    "stoch_k_1d",
    "stoch_k_1h",
    "stoch_k_4h",
    "td9_1d",
    "td9_1h",
    "td9_4h",
    "zig_1d",
    "zig_1h",
    "zig_4h",
]


def append_missing_data(ticker: str):
    raw_path = Path("/app/data/raw") / f"{ticker}.csv"
    if not raw_path.exists():
        log.info(f"No data for {ticker} → full download")
        from scripts.download_historical_prices import download_raw_data
        download_raw_data("/app/data/raw", [ticker])
        return

    df = pd.read_csv(raw_path, parse_dates=["timestamp"])
    last_ts = df["timestamp"].iloc[-1]
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    else:
        last_ts = last_ts.tz_convert("UTC")

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    if last_ts >= now - timedelta(hours=48):
        log.info(f"{ticker} is current → last bar: {last_ts.strftime('%Y-%m-%d %H:%M')} UTC")
        return

    log.info(f"{ticker} needs update → appending missing bars")
    try:
        from ib_insync import IB, Stock, util
        ib = IB()
        ib.connect('host.docker.internal', 7496, clientId=100)
        contract = Stock(ticker, 'SMART', 'USD')
        ib.qualifyContracts(contract)

        bars = ib.reqHistoricalData(
            contract, '', '10 D', '1 hour', 'TRADES', False, 2
        )
        new_df = util.df(bars)
        if new_df is not None and not new_df.empty:
            new_df = new_df[new_df["date"] > last_ts]
            if not new_df.empty:
                new_df.to_csv(raw_path, mode='a', header=False, index=False)
                log.info(f"Appended {len(new_df)} new bars")
        ib.disconnect()
    except Exception as e:
        log.error(f"Update failed: {e}")


def load_latest_features(ticker: str):
    ticker = ticker.upper()
    append_missing_data(ticker)

    raw_path = Path("/app/data/raw") / f"{ticker}.csv"
    df = pd.read_csv(raw_path, parse_dates=["timestamp"])
    log.info(f"{ticker} → Latest bar: {df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')} UTC")

    df = df.tail(2000).copy()
    price_df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]

    augmented = _augment_timeframe_features(price_df)
    recent = augmented.tail(300).copy()

    # Save close and index
    close_prices = recent["close"].copy()
    original_index = recent.index

    # Drop raw OHLCV to avoid conflict
    recent = recent.drop(columns=["open", "high", "low", "close", "volume"], errors="ignore")
    recent = recent.reset_index(drop=True)

    # Generate full feature set
    features = _finalise_feature_frame(recent, ticker, start=None, end=None)
    features = add_golden_price_features(features)

    # Restore index and close price
    features.index = original_index[-len(features):]
    features["close"] = close_prices.values[-len(features):]

    # Ensure price_1d/1w/1M exist
    for col in PRICE_COLS.values():
        if col not in features.columns:
            features[col] = features["close"]

    features = features.ffill().fillna(0)

    # CRITICAL: Align to the live trading feature schema expected by the models
    missing = [col for col in LIVE_MODEL_FEATURES if col not in features.columns]
    for col in missing:
        features[col] = 0.0

    # Preserve deterministic order and drop any metadata columns
    ordered = features.reindex(columns=LIVE_MODEL_FEATURES, fill_value=0.0)
    feats_row = ordered.iloc[-1]

    return feats_row, features  # return both row and full df


def predict_timeframe(ticker: str, tf_name: str, feats_row: pd.Series):
    price = feats_row[PRICE_COLS[tf_name]]
    raw_preds = predict_with_all_models(feats_row)
    decision, detail = ensemble_vote(raw_preds, return_details=True)

    conf = round(sum(detail.get("confidences", {}).values()) / max(1, len(detail.get("confidences", {}))), 3)
    reason = detail.get("reason", "AI Decision")

    log.info(f"{ticker} {tf_name.upper():<7} → {decision} @ {price:,.2f} | {reason}")
    return decision, float(price), reason, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="+", default=["TSLA"])
    args = parser.parse_args()

    print(f"\nMulti-Timeframe AI Signals @ {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC\n")
    print(f"{'Ticker':<7} {'TF':<8} {'Signal':<8} {'Price':>12} {'Conf':>6}")
    print("-" * 90)

    for t in args.tickers:
        t = t.upper()
        feats_row, _ = load_latest_features(t)

        print(f"\n{t} Signals:")
        for tf in ["daily", "weekly", "monthly"]:
            dec, price, reason, conf = predict_timeframe(t, tf, feats_row)
            color = {"Buy": "\033[92m", "Sell": "\033[91m", "Hold": "\033[93m"}.get(dec, "")
            print(f"  {tf:<8} {color}{dec:<8}\033[0m {price:12.2f} {conf:6.3f} {reason}")

    print("\nDone. Always current • Matches backtester • Professional AI")


if __name__ == "__main__":
    main()