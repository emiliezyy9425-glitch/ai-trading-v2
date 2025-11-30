"""Backtester — 100% identical to live trading (features, lookback, decisions, logging)."""

from __future__ import annotations

import argparse
import csv
import logging
import os
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
from ml_predictor import predict_with_all_models, independent_model_decisions
from indicators import summarize_td_sequential
from sp500_above_20d import load_sp500_above_20d_history
from sp500_breadth import calculate_s5tw_history_ibkr_sync
from feature_engineering import default_feature_values

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
PRICE_HISTORY_WINDOW = 200

REQUIRED_FEATURE_COLUMNS: tuple[str, ...] = (
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
)


@dataclass
class Position:
    entry_price: float
    timestamp: datetime


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


def _align_feature_row(raw_features: pd.Series) -> pd.Series:
    defaults = default_feature_values(REQUIRED_FEATURE_COLUMNS)
    aligned = pd.Series(defaults)
    for key, value in raw_features.items():
        aligned[key] = value
    return aligned.reindex(REQUIRED_FEATURE_COLUMNS, fill_value=0.0)


def _compute_price_features(
    price_history: deque[tuple[datetime, float]],
    timestamp: datetime,
    price: float,
    bollinger: dict | None,
    previous_features: pd.Series | None,
) -> dict[str, float]:
    """Return price-derived features using the same formulas as feature_engineering."""

    price_history.append((timestamp, price))
    if len(price_history) > PRICE_HISTORY_WINDOW:
        price_history.popleft()

    dates, prices = zip(*price_history)
    price_series = pd.Series(prices, index=pd.DatetimeIndex(dates))

    returns = np.log(price_series / price_series.shift(1))
    ret_1h = returns.iloc[-1] if not returns.empty else np.nan
    ret_4h = np.log(price_series.iloc[-1] / price_series.shift(4).iloc[-1]) if len(price_series) > 4 else np.nan
    ret_24h = (
        np.log(price_series.iloc[-1] / price_series.shift(24).iloc[-1])
        if len(price_series) > 24
        else np.nan
    )

    rolling_mean = price_series.rolling(window=120, min_periods=60).mean()
    rolling_std = price_series.rolling(window=120, min_periods=60).std()
    price_z = (price_series - rolling_mean) / rolling_std
    price_z_value = price_z.iloc[-1] if not price_z.empty else np.nan

    def _fallback(name: str, value: float | np.floating | None) -> float:
        if value is None or not np.isfinite(float(value)):
            if previous_features is not None:
                prev = previous_features.get(name)
                if pd.notna(prev):
                    return float(prev)
            return 0.0
        return float(value)

    upper = (bollinger or {}).get("upper")
    lower = (bollinger or {}).get("lower")
    bb_position = 0.0
    if upper is not None and lower is not None:
        bb_position = (price - lower) / (upper - lower + 1e-8)
        bb_position = float(np.clip(bb_position, 0.0, 1.0))
    else:
        bb_position = _fallback("bb_position_1h", None)

    price_z_value = _fallback("price_z_120h", price_z_value)
    price_z_value = float(np.clip(price_z_value, -5.0, 5.0))

    return {
        "ret_1h": _fallback("ret_1h", ret_1h),
        "ret_4h": _fallback("ret_4h", ret_4h),
        "ret_24h": _fallback("ret_24h", ret_24h),
        "price_z_120h": price_z_value,
        "bb_position_1h": bb_position,
    }


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

    def _load_raw_bars_from_disk() -> pd.DataFrame:
        raw_dir = Path(PROJECT_ROOT) / "data" / "lake" / "raw" / ticker / timeframe.replace(" ", "_")
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

    def _download_missing_bars_from_ibkr() -> pd.DataFrame:
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
                timeframe,
            )
            df = live_trading.get_historical_data(ib, ticker, timeframe)
            return df if df is not None else pd.DataFrame()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to download IBKR history for %s (%s): %s", ticker, timeframe, exc)
            return pd.DataFrame()
        finally:
            try:
                ib.disconnect()
            except Exception:
                pass

    # Load curated bars first (matches live trading); fall back to locally stored raw prices
    df = live_trading.load_curated_bars(ticker, timeframe)
    if df.empty:
        logger.info("Curated data missing; attempting to use locally stored raw price bars instead.")
        df = _load_raw_bars_from_disk()
    if df.empty:
        df = _download_missing_bars_from_ibkr()
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
    equity_curve = []
    feature_history: deque[pd.Series] = live_trading.FEATURE_HISTORY[ticker]
    price_history: deque[tuple[datetime, float]] = deque(maxlen=PRICE_HISTORY_WINDOW)
    missing_feature_logged = False

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

    logger.info(f"Loaded {len(df)} bars. Running bar-by-bar...")

    def _timeframe_delta(tf: str) -> timedelta:
        tf = tf.strip().lower()
        if tf in {"1 day", "1d", "daily"}:
            return timedelta(days=1)
        if tf in {"4 hours", "4h"}:
            return timedelta(hours=4)
        return timedelta(hours=1)

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
        price_features = _compute_price_features(
            price_history, now, price, indicators.get("bollinger", {}).get(timeframe), previous_features
        )
        raw_features = build_feature_row(
            indicators, price, iv, delta, sp500_pct, previous_features=previous_features
        )
        raw_features.loc["ret_1h"] = price_features["ret_1h"]
        raw_features.loc["ret_4h"] = price_features["ret_4h"]
        raw_features.loc["ret_24h"] = price_features["ret_24h"]
        raw_features.loc["price_z_120h"] = price_features["price_z_120h"]
        raw_features.loc["price_z_120h.1"] = price_features["price_z_120h"]
        raw_features.loc["ret_1h.1"] = price_features["ret_1h"]
        raw_features.loc["ret_4h.1"] = price_features["ret_4h"]
        raw_features.loc["ret_24h.1"] = price_features["ret_24h"]
        raw_features.loc["bb_position_1h"] = price_features["bb_position_1h"]
        raw_features.loc["bb_position_1h.1"] = price_features["bb_position_1h"]

        # Surface any gaps between the live feature builder and the backtest requirements
        missing_cols = [
            col for col in REQUIRED_FEATURE_COLUMNS if col not in raw_features.index
        ]
        if missing_cols and not missing_feature_logged:
            logger.info(
                "Backtest feature row missing %d/%d columns; filling with feature_engineering defaults: %s",
                len(missing_cols),
                len(REQUIRED_FEATURE_COLUMNS),
                ", ".join(missing_cols),
            )
            missing_feature_logged = True

        aligned_features = _align_feature_row(raw_features)
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

        predictions = predict_with_all_models(sequence_df, seq_len=FEATURE_SEQUENCE_WINDOW)
        decision, detail = independent_model_decisions(predictions, return_details=True)

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
        tds_summary = f"{indicators['tds_trend'].get(timeframe, 0)}/{indicators['tds_signal'].get(timeframe, 0)}"
        # TD Sequential summaries are already precomputed per timeframe in the indicator map
        td9_summary = indicators.get("td9_summary", {}).get(timeframe, "")

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

        price_map = indicators.get("price", {})

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
        write_trade_csv({
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "price": round(price, 4),
            "price_1h": _price_value("1 hour"),
            "price_4h": _price_value("4 hours"),
            "price_1d": _price_value("1 day"),
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
    parser.add_argument("--ticker", default="TSLA")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--timeframe", default="1 hour", choices=["1 hour", "4 hours", "1 day"])
    parser.add_argument(
        "--client-id",
        type=int,
        default=None,
        help="Base IBKR client id to avoid conflicts (defaults to 200 or IBKR_BACKTEST_CLIENT_ID env).",
    )
    args = parser.parse_args()

    run_backtest(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
        client_id=args.client_id,
    )


if __name__ == "__main__":
    main()
