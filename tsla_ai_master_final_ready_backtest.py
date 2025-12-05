"""Backtester — 100% identical to live trading (features, lookback, decisions, logging)."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import math
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np

# === EXACT SAME IMPORTS AS LIVE TRADING ===
import tsla_ai_master_final_ready as live_trading
from tsla_ai_master_final_ready import build_feature_row, is_us_equity_session_open
import ml_predictor
from ml_predictor import (
    FEATURE_ALIASES,
    predict_with_all_models,
    ultimate_decision,
)
from indicators import summarize_td_sequential
from sp500_above_20d import load_sp500_above_20d_history
from sp500_breadth import calculate_s5tw_history_ibkr_sync
from feature_engineering import default_feature_values, sanitize_feature_row

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/app")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BACKTEST_TRADE_LOG_PATH = os.path.join(DATA_DIR, "trade_log_backtest.csv")
FEATURE_SEQUENCE_WINDOW = 60
TICKERS_FILE = os.getenv("TICKERS_FILE", "tickers.txt")

REQUIRED_FEATURE_COLUMNS: tuple[str, ...] = (
    "bb_position_1h",
    "bb_position_1h.1",
    "ret_24h",
    "price_z_120h",
    "ret_4h",
    "ret_1h",
    "adx_1h",
    "adx_4h",
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
)

@dataclass
class Position:
    entry_price: float
    timestamp: datetime
    direction: int  # +1 for long, -1 for short
    size: int = 1

    def add(self, price: float, timestamp: datetime):
        total_cost = (self.entry_price * self.size) + price
        self.size += 1
        self.entry_price = total_cost / self.size
        self.timestamp = timestamp

    def pnl(self, exit_price: float) -> float:
        return (exit_price - self.entry_price) * self.size * self.direction


@contextmanager
def _backtest_time_alignment(reference: datetime):
    """Align live trading timestamp helpers to the simulated bar time."""

    original = live_trading._last_completed_bar_timestamp

    def _patched(timeframe: str = "1 hour", ref: datetime | None = None) -> datetime:
        return original(timeframe, reference)

    live_trading._last_completed_bar_timestamp = _patched
    try:
        yield
    finally:
        live_trading._last_completed_bar_timestamp = original


def write_trade_csv(row: dict):
    """Append a row to the backtest log with proper header handling."""
    file_exists = os.path.isfile(BACKTEST_TRADE_LOG_PATH)
    with open(BACKTEST_TRADE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            logger.info("Backtest trade log columns: %s", ", ".join(row.keys()))
            writer.writeheader()
        writer.writerow(row)


def _align_feature_row(
    raw_features: pd.Series, previous_features: pd.Series | None = None
) -> pd.Series:
    """Align a raw feature row to the exact schema with sensible fallbacks."""

    defaults = default_feature_values(REQUIRED_FEATURE_COLUMNS)
    aligned = pd.Series(defaults)

    if previous_features is not None:
        for col in REQUIRED_FEATURE_COLUMNS:
            prev_val = previous_features.get(col)
            if pd.notna(prev_val):
                aligned[col] = prev_val

    for key, value in raw_features.items():
        aligned[key] = value

    for alias, source in FEATURE_ALIASES.items():
        if source not in aligned:
            continue

        alias_value = aligned.get(alias)
        if alias not in aligned or pd.isna(alias_value) or alias_value == 0:
            aligned[alias] = aligned[source]

    aligned = aligned.reindex(REQUIRED_FEATURE_COLUMNS, fill_value=0.0)
    return sanitize_feature_row(aligned, REQUIRED_FEATURE_COLUMNS)


def _load_tickers(path: str = TICKERS_FILE) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        logger.error("Tickers file not found: %s", file_path)
        return []

    tickers: list[str] = []
    with file_path.open(encoding="utf-8") as f:
        for line in f:
            ticker = line.strip()
            if not ticker or ticker.startswith("#"):
                continue
            tickers.append(ticker)

    if not tickers:
        logger.error("No tickers found in %s", file_path)

    return tickers


def run_backtest(
    ticker: str = "TSLA",
    start_date: str = "2024-01-01",
    end_date: str | None = None,
    timeframe: str = "1 hour",
    client_id: int | None = None,
):
    if end_date is None:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)

    logger.info(f"Starting EXACT-MATCH backtest: {ticker} | {start_date} → {end_date} | {timeframe}")

    base_client_id = client_id or int(os.getenv("IBKR_BACKTEST_CLIENT_ID", "200"))

    def _load_raw_bars_from_disk(tf: str = timeframe) -> pd.DataFrame:
        raw_dir = Path(PROJECT_ROOT) / "data" / "lake" / "raw" / ticker / tf.replace(" ", "_")
        if not raw_dir.is_dir():
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for path in sorted(raw_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(path)
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
                frames.append(df.set_index("timestamp"))
            except Exception as exc:
                logger.warning(f"Failed to load raw parquet {path}: {exc}")

        for path in sorted(raw_dir.glob("*.csv")):
            try:
                frames.append(
                    pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
                )
            except Exception as exc:
                logger.warning(f"Failed to load raw CSV {path}: {exc}")

        if not frames:
            return pd.DataFrame()

        merged = pd.concat(frames)
        merged = merged[~merged.index.duplicated(keep="last")]
        merged.index = pd.to_datetime(merged.index, utc=True)
        return merged.sort_index()

    def _load_backtest_bars(tf: str) -> pd.DataFrame:
        """Fetch OHLCV data for a timeframe with curated → raw → IBKR fallbacks."""

        curated = live_trading.load_curated_bars(ticker, tf)
        if not curated.empty:
            return curated

        raw = _load_raw_bars_from_disk(tf)
        if not raw.empty:
            logger.info("Using raw %s bars for %s because curated data is missing.", tf, ticker)
            return raw

        downloaded = _download_missing_bars_from_ibkr(tf)
        if not downloaded.empty:
            logger.info(
                "Downloaded %s bars from IBKR for %s because curated/raw data was unavailable.",
                tf,
                ticker,
            )
        return downloaded

    def _download_missing_bars_from_ibkr(tf: str = timeframe) -> pd.DataFrame:
        """Connect to IBKR and download historical bars when local data is missing."""

        try:
            ib = live_trading.connect_ibkr(max_retries=1, initial_client_id=base_client_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("IBKR connection attempt failed: %s", exc)
            return pd.DataFrame()

        if ib is None:
            logger.warning("IBKR connection unavailable; cannot download price history.")
            return pd.DataFrame()

        try:
            logger.info(
                "Attempting IBKR download for %s (%s) because curated/raw data is missing...",
                ticker,
                tf,
            )
            df = live_trading.get_historical_data(ib, ticker, tf)
            return df if df is not None else pd.DataFrame()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to download IBKR history for %s (%s): %s", ticker, tf, exc)
            return pd.DataFrame()
        finally:
            try:
                ib.disconnect()
            except Exception:
                pass

    # Load curated bars first (matches live trading); fall back to locally stored raw prices
    df = _load_backtest_bars(timeframe)
    if df.empty:
        raise FileNotFoundError(
            "No curated, raw, or IBKR-downloaded price data found. "
            "Ensure IBKR is reachable or place raw bars under "
            "data/lake/raw/<TICKER>/<timeframe_with_underscores>/"
        )

    # Load historical breadth data once so we can align it with backtest bars
    history_end_date = (end_dt - timedelta(days=1)).date()
    s5tw_history = pd.Series(dtype=float)

    try:
        ib_for_breadth = live_trading.connect_ibkr(
            max_retries=1, initial_client_id=base_client_id
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("IBKR connection for breadth history failed: %s", exc)
        ib_for_breadth = None

    if ib_for_breadth is not None:
        try:
            s5tw_history = calculate_s5tw_history_ibkr_sync(
                ib_for_breadth,
                start_dt.date(),
                history_end_date,
            )
        except Exception as exc:  # pragma: no cover - IBKR/backfill errors
            logger.warning(
                "IBKR breadth history unavailable (%s); falling back to CSV export.",
                exc,
            )
        finally:
            try:
                ib_for_breadth.disconnect()
            except Exception:
                pass

    if s5tw_history.empty:
        breadth_path = Path(DATA_DIR) / "S&P 500 Stocks Above 20-Day Average Historical Data.csv"
        s5tw_history = load_sp500_above_20d_history(breadth_path)

    if not s5tw_history.empty:
        s5tw_history = s5tw_history.sort_index()

    position: Position | None = None
    peak_price: float | None = None
    equity_curve = []
    feature_history: deque[pd.Series] = live_trading.FEATURE_HISTORY[ticker]

    if not feature_history:
        try:
            live_trading._seed_feature_history_from_historical_data(ticker, None)
        except Exception as exc:  # pragma: no cover - defensive seeding
            logger.warning("Feature history seed failed for %s: %s", ticker, exc)
        feature_history = live_trading.FEATURE_HISTORY[ticker]

    if feature_history:
        logger.info(
            "Seeded %d/%d feature rows before backtest loop.",
            len(feature_history),
            FEATURE_SEQUENCE_WINDOW,
        )
    else:
        logger.info("No historical seed available; will backfill defaults to 60 rows.")

    # Trim to the requested window
    df = df[(df.index >= start_dt) & (df.index < end_dt)].sort_index()

    if df.empty:
        logger.error("No data in date range")
        return

    # Cache higher timeframes for price-aware feature calculations
    price_frames: dict[str, pd.DataFrame] = {}
    for tf in ("4 hours", "1 day"):
        tf_df = _load_backtest_bars(tf)

        if tf_df.empty:
            logger.warning("No %s price data available for %s; features may degrade.", tf, ticker)
            continue

        tf_df = tf_df.sort_index()
        tf_df.index = pd.to_datetime(tf_df.index, utc=True)
        price_frames[tf] = tf_df

    logger.info(f"Loaded {len(df)} bars. Running bar-by-bar...")

    def _timeframe_delta(tf: str) -> timedelta:
        tf = tf.strip().lower()
        if tf in {"1 day", "1d", "daily"}:
            return timedelta(days=1)
        if tf in {"4 hours", "4h"}:
            return timedelta(hours=4)
        return timedelta(hours=1)

    def _ema10_change_from_frame(frame: pd.DataFrame, tf: str, cutoff: datetime) -> float | None:
        if frame.empty or "close" not in frame:
            return None

        history = frame[frame.index <= cutoff]
        if len(history) < 2:
            return None

        if tf == "1 day":
            ema_series = history["close"].ewm(span=10, adjust=False).mean()
        else:
            daily_close = history["close"].resample("1D").last()
            ema_series = daily_close.ewm(span=10, adjust=False).mean()
            ema_series = ema_series.reindex(history.index, method="ffill")

        if len(ema_series) < 2:
            return None

        ema_last = ema_series.iloc[-1]
        ema_prev = ema_series.iloc[-2]

        if any(pd.isna(val) for val in (ema_last, ema_prev)):
            return None

        return float(ema_last - ema_prev)

    for bar_idx, (ts, bar) in enumerate(df.iterrows()):
        now = ts.to_pydatetime()
        reference = now + _timeframe_delta(timeframe)

        # Only trade during US equity session (same as live)
        if not is_us_equity_session_open(now):
            continue

        try:
            with _backtest_time_alignment(reference):
                indicators = live_trading.get_multi_timeframe_indicators(None, ticker)
        except Exception as e:
            logger.warning(f"Indicator build failed at {now}: {e}")
            continue

        ema_change_map = indicators.setdefault("ema10_change", {})
        frame_map = {timeframe: df, **price_frames}

        for tf, frame in frame_map.items():
            if frame.empty:
                continue

            last_completed = frame.index[frame.index <= now]
            if last_completed.empty:
                continue

            cutoff = last_completed[-1].to_pydatetime()
            ema_delta = _ema10_change_from_frame(frame, tf, cutoff)
            if ema_delta is None or math.isnan(ema_delta):
                continue

            ema_change_map[tf] = ema_delta

        # === PRICE RECONSTRUCTION (price-free system) ===
        if "close" in bar:
            try:
                price = float(bar["close"])
            except (TypeError, ValueError):
                price = float("nan")
        elif "price_1h" in bar:
            price = float(bar["price_1h"])
        elif "ret_1h" in df.columns:
            cum_ret = df["ret_1h"].iloc[: bar_idx + 1].sum()
            price = round(float(np.exp(cum_ret) * 100.0), 4)
        else:
            price = 100.0
        # ================================================
        if not np.isfinite(price):
            price = price_history[-1][1] if price_history else 100.0

        if position is None:
            peak_price = None

        if position is not None:
            if peak_price is None:
                peak_price = position.entry_price

            # Update peak price for trailing stop (runs every bar)
            if position.direction > 0 and price > peak_price:
                peak_price = price
            if position.direction < 0 and price < peak_price:
                peak_price = price

            # 18% trailing stop — checked EVERY bar, even on HOLD
            position_direction_was_long = position.direction > 0
            drawdown = (peak_price - price) / peak_price if position.direction > 0 else (price - peak_price) / peak_price
            if drawdown >= 0.18:
                pnl_points = (price - position.entry_price) * position.direction
                equity_curve.append(pnl_points)
                result = "STOP_LOSS"
                trade_size = position.size
                position = None
                peak_price = None
                decision = "SELL" if position_direction_was_long else "BUY"
                logger.info(f"TRAILING STOP HIT → {result} {trade_size}x at {price}")
                continue  # skip rest of loop
        iv = indicators.get("iv", 50.0)
        delta = indicators.get("delta", 0.5)
        sp500_pct = indicators.get("sp500_above_20d", 50.0)
        if not s5tw_history.empty:
            lookup_date = now.date()
            if lookup_date in s5tw_history.index:
                sp500_pct = float(s5tw_history.loc[lookup_date])
            else:
                historical = s5tw_history.loc[:lookup_date]
                if not historical.empty:
                    sp500_pct = float(historical.iloc[-1])

        # EXACT SAME FEATURE ROW AS LIVE TRADING
        previous_features = feature_history[-1] if feature_history else None
        raw_features = build_feature_row(
            indicators, price, iv, delta, sp500_pct, previous_features=previous_features
        )
        aligned_features = _align_feature_row(raw_features, previous_features)
        feature_history.append(aligned_features)

        sequence_df = pd.DataFrame(feature_history).reindex(
            columns=REQUIRED_FEATURE_COLUMNS, fill_value=0.0
        )
        if len(sequence_df) < FEATURE_SEQUENCE_WINDOW:
            padding_needed = FEATURE_SEQUENCE_WINDOW - len(sequence_df)
            padding = pd.DataFrame(
                [default_feature_values(REQUIRED_FEATURE_COLUMNS)] * padding_needed
            )
            sequence_df = pd.concat([padding, sequence_df], ignore_index=True)
        sequence_df.index = pd.date_range(
            end=now, periods=len(sequence_df), freq=_timeframe_delta(timeframe)
        )

        preds, ppo_meta = predict_with_all_models(
            sequence_df, seq_len=FEATURE_SEQUENCE_WINDOW
        )

        # === NEW: Use the correct ultimate decision engine ===
        decision_state, direction, reason = ml_predictor.ultimate_decision(
            preds, ppo_meta
        )
        decision_upper = direction.upper() if decision_state == "EXECUTE" else "HOLD"
        print(f"DEBUG: {now} → {decision_state} {direction} | {reason}")

        # For logging compatibility
        detail = {
            "reason": reason,
            "decision_state": decision_state,
            "direction": direction,
            "ppo_action": ppo_meta.get("action", 1),
            "ppo_value": ppo_meta.get("value", 0.0),
            "ppo_value_ma100": ppo_meta.get("value_ma100", 0.0),
            "ppo_value_std100": ppo_meta.get("value_std100", 0.5),
            "ppo_entropy": ppo_meta.get("entropy", 0.0),
            "votes": {
                k: ("Buy" if p[-1] > 0.5 else "Sell")
                for k, (p, _) in preds.items()
                if k != "PPO" and len(p)
            },
            "confidences": {
                k: max(float(pr[-1]), 1 - float(pr[-1]))
                for k, (pr, _) in preds.items()
                if k != "PPO" and len(pr)
            },
            "probabilities": {
                k: float(pr[-1]) if len(pr) else 0.5
                for k, (pr, _) in preds.items()
                if k != "PPO"
            },
        }

        if "PPO" in preds:
            ppo_prob, _ = preds.get("PPO", ([], []))
            if len(ppo_prob):
                detail.setdefault("votes", {})["PPO"] = (
                    "Buy" if ppo_prob[-1] > 0.5 else "Sell"
                )
                detail.setdefault("confidences", {})["PPO"] = max(
                    float(ppo_prob[-1]), 1 - float(ppo_prob[-1])
                )
                detail.setdefault("probabilities", {})["PPO"] = float(ppo_prob[-1])

        decision = decision_upper

        # === CORRECTED POSITION MANAGEMENT (paste this exactly) ===
        signal_triggered = False
        trade_filled = False
        pnl_points = 0.0
        result = "HOLD"
        trade_size = position.size if position is not None else 0
        decision = decision.upper()

        if detail.get("decision_state") == "EXECUTE":
            signal_triggered = True
            expected_direction = 1 if decision == "BUY" else -1

            # Open or add to position
            if position is None:
                position = Position(
                    entry_price=price,
                    timestamp=now,
                    direction=expected_direction,
                    size=1
                )
                peak_price = price
                trade_filled = True
                trade_size = position.size
                result = "ENTRY"
                equity_curve.append(0)  # first bar = no PnL yet
            else:
                # Already in correct direction → pyramid or just hold
                if position.direction == expected_direction:
                    position.size += 1
                    trade_filled = True  # count as additional fill
                    trade_size = position.size
                    result = "PYRAMID"
                # Reverse signal → close old, open new
                else:
                    # Close old position
                    pnl_points = (price - position.entry_price) * position.direction
                    equity_curve.append(pnl_points)
                    result = "WIN" if pnl_points > 0 else "LOSS" if pnl_points < 0 else "FLAT"
                    trade_size = position.size
                    # Open new position
                    position = Position(
                        entry_price=price,
                        timestamp=now,
                        direction=expected_direction,
                        size=1
                    )
                    peak_price = price
                    trade_filled = True
        else:
            # === NEW RULE: HOLD THROUGH "HOLD" SIGNAL ===
            # We do NOTHING when nuclear signal disappears
            # Position stays 100% open with all pyramids
            decision = "HOLD"
            # ← signal weakened, but we ride the trend
            result = "HOLD"                 # ← no exit yet
            pnl = 0.0
            signal_triggered = False
            trade_filled = False

        pnl = pnl_points
        decision_upper = decision.upper()

        # Human-readable summaries
        tds_summary = f"{indicators['tds_trend'].get(timeframe, 0)}/{indicators['tds_signal'].get(timeframe, 0)}"
        # TD Sequential summaries are already precomputed per timeframe in the indicator map
        td9_summary = indicators.get("td9_summary", {}).get(timeframe, "")

        # FULL MODEL TRANSPARENCY
        detail = detail if isinstance(detail, dict) else {}

        # === Extract real model votes & confidences from the new ensemble ===
        votes = detail.get("votes", {})
        confs = detail.get("confidences", {})

        rf_v = votes.get("RandomForest", "Hold")
        rf_c = confs.get("RandomForest", 0.0)
        xgb_v = votes.get("XGBoost", "Hold")
        xgb_c = confs.get("XGBoost", 0.0)
        lgb_v = votes.get("LightGBM", "Hold")
        lgb_c = confs.get("LightGBM", 0.0)
        lstm_v = votes.get("LSTM", "Hold")
        lstm_c = confs.get("LSTM", 0.0)
        trans_v = votes.get("Transformer", "Hold")
        trans_c = confs.get("Transformer", 0.0)

        # TCN — safe defaults using actual prediction probabilities
        tcn_prob = preds.get("TCN", (np.array([0.5]), np.array([0])))[0]
        tcn_prob = tcn_prob[0] if len(tcn_prob) > 0 else 0.5
        tcn_v = "Buy" if tcn_prob > 0.5 else "Sell"
        tcn_c = tcn_prob if tcn_prob > 0.5 else 1 - tcn_prob

        ppo_v = "Aggressive" if detail.get("ppo_action") == 2 else "Reduce" if detail.get("ppo_action") == 0 else "Hold"
        ppo_c = detail.get("ppo_value", 0.0)

        price_map = indicators.get("price", {})

        for tf, frame in price_frames.items():
            last_completed = live_trading._last_completed_bar_timestamp(tf, reference)
            tf_price = frame[frame.index <= last_completed]["close"]
            if tf_price.empty:
                continue
            price_map[tf] = float(tf_price.iloc[-1])

        price_map[timeframe] = price
        indicators["price"] = price_map

        def _price_value(key: str) -> float | str:
            val = price_map.get(key)
            try:
                return round(float(val), 3)
            except (TypeError, ValueError):
                return ""

        def _serialize_feature_value(val):
            if pd.isna(val):
                return ""
            if isinstance(val, (float, int, bool, str)):
                return val
            try:
                return float(val)
            except Exception:
                return str(val)

        feature_logs = {
            f"feature_{k}": _serialize_feature_value(v) for k, v in aligned_features.items()
        }

        # Write full audit trail
        current_size = 0 if position is None else position.size
        write_trade_csv({
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "price": round(price, 4),
            "price_1h": _price_value("1 hour"),
            "price_4h": _price_value("4 hours"),
            "price_1d": _price_value("1 day"),
            "decision": decision_upper,
            "result": result,
            "pnl": round(pnl, 3),
            "position_size": current_size if decision_upper != "SELL" else trade_size,
            "executed": "Yes" if signal_triggered else "No",
            "trade_filled": "Yes" if trade_filled else "No",

            # Ensemble
            "ensemble_reason": detail.get("reason", ""),

            # Individual model votes & confidences
            "rf_vote": rf_v, "rf_conf": round(rf_c, 4) if rf_c else 0.0,
            "xgb_vote": xgb_v, "xgb_conf": round(xgb_c, 4) if xgb_c else 0.0,
            "lgb_vote": lgb_v, "lgb_conf": round(lgb_c, 4) if lgb_c else 0.0,
            "lstm_vote": lstm_v, "lstm_conf": round(lstm_c, 4) if lstm_c else 0.0,
            "tcn_vote": tcn_v,
            "tcn_conf": round(tcn_c, 4) if tcn_c else 0.0,
            "ppo_vote": ppo_v, "ppo_conf": round(detail.get("confidences", {}).get("PPO", 0.0), 4),
            "transformer_vote": trans_v, "transformer_conf": round(trans_c, 4) if trans_c else 0.0,
            "ppo_action_num": detail.get("ppo_action", 1),

            # Human summaries
            "tds_summary": tds_summary,
            "td9_summary": td9_summary,
            "rsi": round(indicators["rsi"].get(timeframe, 0), 2),
            "macd": round(indicators["macd"].get(timeframe, 0), 4),
            "volume": int(indicators["volume"].get(timeframe, 0)),
            "iv": round(iv, 2),
            **feature_logs,
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
    parser.add_argument(
        "--ticker",
        default=None,
        help="Run a single ticker. If omitted, tickers are loaded from tickers.txt.",
    )
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--timeframe", default="1 hour", choices=["1 hour", "4 hours", "1 day"])
    parser.add_argument(
        "--client-id",
        type=int,
        default=None,
        help="Base IBKR client id to avoid conflicts (defaults to 200 or IBKR_BACKTEST_CLIENT_ID env).",
    )
    parser.add_argument(
        "--tickers-file",
        default=TICKERS_FILE,
        help="Path to ticker list file (one ticker per line). Ignored when --ticker is set.",
    )
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else _load_tickers(args.tickers_file)

    if not tickers:
        logger.error("No tickers to backtest. Exiting.")
        return

    for ticker in tickers:
        logger.info("Starting backtest for %s", ticker)
        run_backtest(
            ticker=ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            timeframe=args.timeframe,
            client_id=args.client_id,
        )


if __name__ == "__main__":
    main()
