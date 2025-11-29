"""Backtester — 100% identical to live trading (features, lookback, decisions, logging)."""

from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

# === EXACT SAME IMPORTS AS LIVE TRADING ===
from tsla_ai_master_final_ready import (
    get_multi_timeframe_indicators,
    build_feature_row,
    is_us_equity_session_open,
)
from ml_predictor import predict_with_all_models, ensemble_vote
from indicators import summarize_td_sequential

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/app")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BACKTEST_TRADE_LOG_PATH = os.path.join(DATA_DIR, "trade_log_backtest.csv")


@dataclass
class Position:
    entry_price: float
    timestamp: datetime


def write_trade_csv(row: dict):
    """Append a row to the backtest log with proper header handling."""
    file_exists = os.path.isfile(BACKTEST_TRADE_LOG_PATH)
    with open(BACKTEST_TRADE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_backtest(
    ticker: str = "TSLA",
    start_date: str = "2024-01-01",
    end_date: str | None = None,
    timeframe: str = "1 hour",
):
    if end_date is None:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)

    logger.info(f"Starting EXACT-MATCH backtest: {ticker} | {start_date} → {end_date} | {timeframe}")

    # Verify curated data exists (this is what live uses)
    parquet_path = os.path.join(PROJECT_ROOT, "data", "lake", "curated", f"{ticker}_{timeframe.replace(' ', '')}.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Real curated data missing: {parquet_path}\n"
            "Run: python /app/scripts/compute_historical_indicators.py --ticker TSLA --years 3"
        )

    position: Position | None = None
    equity_curve = []

    # Load full history once
    df = pd.read_parquet(parquet_path)
    df = df[(df.index >= start_dt) & (df.index < end_dt)].sort_index()

    if df.empty:
        logger.error("No data in date range")
        return

    logger.info(f"Loaded {len(df)} bars. Running bar-by-bar...")

    for ts, _ in df.iterrows():
        now = ts.to_pydatetime()

        # Only trade during US equity session (same as live)
        if not is_us_equity_session_open(now):
            continue

        try:
            indicators = get_multi_timeframe_indicators(ticker, now)
        except Exception as e:
            logger.warning(f"Indicator build failed at {now}: {e}")
            continue

        price = indicators["price"][timeframe]
        iv = indicators.get("iv", 50.0)
        delta = indicators.get("delta", 0.5)
        sp500_pct = indicators.get("sp500_above_20d", 50.0)

        # EXACT SAME FEATURE ROW AS LIVE TRADING
        features = build_feature_row(indicators, price, iv, delta, sp500_pct)
        features_df = pd.DataFrame([features])

        predictions = predict_with_all_models(features_df)
        decision, detail = ensemble_vote(predictions, return_details=True)

        # Handle position logic
        executed = False
        pnl = 0.0
        result = "HOLD"

        if decision == "Buy" and position is None:
            position = Position(entry_price=price, timestamp=now)
            executed = True
            result = "ENTRY"
        elif decision == "Sell" and position is not None:
            pnl = price - position.entry_price
            result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT"
            executed = True
            equity_curve.append(pnl)
            position = None

        # Human-readable summaries
        fib_summary = indicators["fib_summary"].get(timeframe, "")
        tds_summary = f"{indicators['tds_trend'].get(timeframe, 0)}/{indicators['tds_signal'].get(timeframe, 0)}"
        td9_summary = summarize_td_sequential(indicators.get("td9_summary", {})).get(timeframe, "")

        # FULL MODEL TRANSPARENCY
        detail = detail if isinstance(detail, dict) else {}
        votes = detail.get("votes", {})
        confs = detail.get("confidences", {})
        ignored = detail.get("ignored_models", {})
        missing = detail.get("missing_models", {})

        def get_vote_conf(name: str):
            if name in missing:
                return "MISSING", 0.0
            if name in ignored:
                return "ERROR", 0.0
            vote = votes.get(name, "")
            conf = confs.get(name, 0.0)
            return vote, conf

        rf_v, rf_c = get_vote_conf("RandomForest")
        xgb_v, xgb_c = get_vote_conf("XGBoost")
        lgb_v, lgb_c = get_vote_conf("LightGBM")
        lstm_v, lstm_c = get_vote_conf("LSTM")
        ppo_v, ppo_c = get_vote_conf("PPO")
        trans_v, trans_c = get_vote_conf("Transformer")

        # Write full audit trail
        write_trade_csv({
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "price": round(price, 3),
            "decision": decision.upper(),
            "result": result,
            "pnl": round(pnl, 3),
            "executed": "Yes" if executed else "No",

            # Ensemble
            "ensemble_reason": detail.get("reason", ""),

            # Individual model votes & confidences
            "rf_vote": rf_v, "rf_conf": round(rf_c, 4),
            "xgb_vote": xgb_v, "xgb_conf": round(xgb_c, 4),
            "lgb_vote": lgb_v, "lgb_conf": round(lgb_c, 4),
            "lstm_vote": lstm_v, "lstm_conf": round(lstm_c, 4),
            "ppo_vote": ppo_v, "ppo_conf": round(ppo_c, 4),
            "transformer_vote": trans_v, "transformer_conf": round(trans_c, 4),

            # Human summaries
            "fib_summary": fib_summary,
            "tds_summary": tds_summary,
            "td9_summary": td9_summary,
            "rsi": round(indicators["rsi"].get(timeframe, 0), 2),
            "macd": round(indicators["macd"].get(timeframe, 0), 4),
            "volume": int(indicators["volume"].get(timeframe, 0)),
            "iv": round(iv, 2),
        })

    # Final stats
    total = len(equity_curve)
    wins = len([x for x in equity_curve if x > 0])
    win_rate = wins / total * 100 if total else 0
    total_pnl = sum(equity_curve)

    logger.info("BACKTEST COMPLETE")
    logger.info(f"Total trades : {total}")
    logger.info(f"Win rate     : {win_rate:.1f}%")
    logger.info(f"Total PnL    : {total_pnl:+.2f} points")
    logger.info(f"Log saved to : {BACKTEST_TRADE_LOG_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Exact-match backtester")
    parser.add_argument("--ticker", default="TSLA")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--timeframe", default="1 hour", choices=["1 hour", "4 hours", "1 day"])
    args = parser.parse_args()

    run_backtest(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
    )


if __name__ == "__main__":
    main()