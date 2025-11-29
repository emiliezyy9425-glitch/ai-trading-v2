"""
Generate weekly, daily, and monthly predictions for a ticker using existing models.

This script loads curated daily bars from ``data/lake/curated/<TICKER>_1_day``
and derives weekly and monthly aggregates. It builds a single feature row per
requested timeframe, feeds it through the LSTM and Transformer models, and prints
out the resulting direction (Buy/Sell/Hold).

Usage:
    python timeframe_predictions.py --ticker TSLA
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from indicators import (
    calculate_adx,
    calculate_obv,
    calculate_stochastic_oscillator,
)
from ml_predictor import (
    FEATURE_NAMES,
    CANDLESTICK_FEATURES,
    ADDITIONAL_FEATURES,
    predict_with_all_models,
    independent_model_decisions,
)
from project_paths import resolve_data_path
from scripts.compute_historical_indicators import compute_indicators
from self_learn import _BASE_FEATURES  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SUPPORTED_SUFFIXES: tuple[str, ...] = ("1h", "4h", "1d")


def _load_daily_prices(ticker: str) -> pd.DataFrame:
    """Load curated daily bars for *ticker*.

    The helper mirrors ``load_curated_bars`` but avoids heavy dependencies by
    checking for either parquet or CSV files under ``data/lake/curated``.
    """

    base = resolve_data_path("lake", "curated", f"{ticker}_1_day")
    candidates: Iterable[Path] = (base.with_suffix(".parquet"), base.with_suffix(".csv"))

    for path in candidates:
        if path.exists():
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index, utc=True)
            return df.sort_index()

    raise FileNotFoundError(
        f"No curated daily data found for {ticker}. Expected {base}.parquet or {base}.csv"
    )


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(how="any")
    )


def _compute_changes(series: pd.Series) -> float:
    if len(series) < 2:
        return 0.0
    delta = series.iloc[-1] - series.iloc[-2]
    if pd.isna(delta):
        return 0.0
    return float(delta)


def _build_base_features(df: pd.DataFrame, timeframe: str) -> Dict[str, float]:
    """Compute a minimal set of base indicators for the given timeframe."""

    indicators = compute_indicators(df, timeframe)
    if indicators.empty:
        raise ValueError(f"No indicators available for timeframe {timeframe}")

    latest = indicators.tail(2)
    base: Dict[str, float] = {}

    base["rsi"] = float(latest["rsi"].iloc[-1])
    base["macd"] = float(latest["macd"].iloc[-1])
    base["signal"] = float(latest["signal"].iloc[-1])
    base["ema10"] = float(latest["ema10"].iloc[-1])
    base["price_above_ema10"] = float(latest["price_above_ema10"].iloc[-1])
    base["ema10_dev"] = float(latest["ema10_dev"].iloc[-1])
    base["bb_upper"] = float(latest["bb_upper"].iloc[-1])
    base["bb_lower"] = float(latest["bb_lower"].iloc[-1])
    base["bb_mid"] = float(latest["bb_mid"].iloc[-1])
    base["volume"] = float(df["volume"].iloc[-1]) if "volume" in df else 0.0
    base["atr"] = float(latest["atr"].iloc[-1])

    # Derived changes
    base["rsi_change"] = _compute_changes(latest["rsi"])
    base["macd_change"] = _compute_changes(latest["macd"])
    base["ema10_change"] = _compute_changes(latest["ema10"])

    # Trend and volume-derived metrics (best-effort; fallback to zero)
    adx_value = calculate_adx(df)
    if isinstance(adx_value, pd.Series):
        base["adx"] = float(adx_value.iloc[-1]) if len(adx_value) else 0.0
    else:
        base["adx"] = float(adx_value or 0.0)

    obv_value = calculate_obv(df)
    if isinstance(obv_value, pd.Series):
        base["obv"] = float(obv_value.iloc[-1]) if len(obv_value) else 0.0
    else:
        base["obv"] = float(obv_value or 0.0)

    stoch_k, stoch_d = calculate_stochastic_oscillator(df["high"], df["low"], df["close"])
    if isinstance(stoch_k, pd.Series) and len(stoch_k):
        base["stoch_k"] = float(stoch_k.iloc[-1])
    else:
        base["stoch_k"] = float(stoch_k or 0.0)
    if isinstance(stoch_d, pd.Series) and len(stoch_d):
        base["stoch_d"] = float(stoch_d.iloc[-1])
    else:
        base["stoch_d"] = float(stoch_d or 0.0)

    # Placeholders for features not derivable from price alone
    for placeholder in [
        "tds_trend",
        "tds_signal",
        "high_vol",
        "vol_spike",
        "td9",
        "zig",
        "vol_cat",
        "fib_level1",
        "fib_level2",
        "fib_level3",
        "fib_level4",
        "fib_level5",
        "fib_level6",
        "fib_time_count",
        "fib_zone_delta",
    ]:
        base.setdefault(placeholder, 0.0)

    return base


def _assemble_feature_row(base_features: Dict[str, float]) -> pd.DataFrame:
    """Map base indicators into the trained feature schema."""

    features: Dict[str, float] = {}
    for suffix in SUPPORTED_SUFFIXES:
        for name in _BASE_FEATURES:
            features[f"{name}_{suffix}"] = float(base_features.get(name, 0.0))

    for name in ADDITIONAL_FEATURES:
        features[name] = 0.0

    for pattern in CANDLESTICK_FEATURES:
        features[pattern] = 0.0

    frame = pd.DataFrame([features]).reindex(columns=FEATURE_NAMES, fill_value=0.0)
    frame.index = pd.DatetimeIndex([pd.Timestamp.utcnow()])
    return frame


def _predict_for_frame(feature_frame: pd.DataFrame) -> str:
    model_outputs = predict_with_all_models(feature_frame)
    decision, detail = independent_model_decisions(model_outputs, return_details=True)
    logger.info("Model details: %s", detail)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly/Daily/Monthly predictions")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g., TSLA")
    args = parser.parse_args()

    prices = _load_daily_prices(args.ticker)

    timeframes = {
        "daily": prices,
        "weekly": _resample(prices, "W"),
        "monthly": _resample(prices, "ME"),
    }

    results: Dict[str, str] = {}
    for label, frame in timeframes.items():
        if frame.empty:
            logger.warning("No data for %s timeframe", label)
            results[label] = "No Data"
            continue
        base = _build_base_features(frame, timeframe=label)
        features = _assemble_feature_row(base)
        results[label] = _predict_for_frame(features)

    for label, decision in results.items():
        print(f"{label.capitalize()} timeframe prediction: {decision}")


if __name__ == "__main__":
    main()
