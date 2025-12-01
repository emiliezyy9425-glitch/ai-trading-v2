"""Minimal feature builder used by the live prediction script.

This module intentionally mirrors the indicator formulas used during training
so the live pipeline can compute every model input directly from the latest
OHLCV bars. The functions here are lightweight but self contained so they can
be imported without pulling in the full trading agent.
"""

from __future__ import annotations

import pandas as pd

from feature_engineering import default_feature_values
from indicators import (
    calculate_adx,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_rsi,
    calculate_stochastic_oscillator,
    calculate_td_sequential,
    calculate_zig_zag,
    fast_candlestick_patterns,
    summarize_td_sequential,
)
from feature_engineering import encode_td9, encode_zig


def append_historical_data(path):
    return False


def generate_historical_data(path):
    return False


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Return OHLCV resampled to ``rule`` using training-time aggregation."""

    if df is None or df.empty:
        return pd.DataFrame()

    return (
        df.resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(how="all")
    )


def _compute_timeframe_features(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Compute core indicator features for a single timeframe."""

    if df.empty:
        return pd.DataFrame(index=df.index)

    current_index = df.index
    rsi_val = calculate_rsi(df) or 0.0
    rsi_prev = calculate_rsi(df.iloc[:-1]) if len(df) > 1 else None
    rsi_change = (rsi_val - rsi_prev) if rsi_prev is not None else 0.0

    macd_val, signal_val = calculate_macd(df)
    macd_val = macd_val or 0.0
    signal_val = signal_val or 0.0
    macd_prev, _ = calculate_macd(df.iloc[:-1]) if len(df) > 1 else (None, None)
    macd_change = (macd_val - macd_prev) if macd_prev is not None else 0.0

    ema_series = df["close"].ewm(span=10, adjust=False).mean()
    ema10 = ema_series.iloc[-1] if not ema_series.empty else 0.0
    ema_prev = ema_series.iloc[-2] if len(ema_series) > 1 else None
    price_last = df["close"].iloc[-1]
    ema10_dev = (price_last - ema10) / ema10 if ema10 else 0.0
    ema10_change = (ema10 - ema_prev) if ema_prev is not None else 0.0
    price_above = 1 if price_last > ema10 else 0

    adx_val = calculate_adx(df) or 0.0
    stoch_vals = calculate_stochastic_oscillator(df)
    if isinstance(stoch_vals, tuple):
        stoch_k, stoch_d = stoch_vals
    else:
        stoch_k, stoch_d = 0.0, 0.0
    stoch_k = stoch_k or 0.0
    stoch_d = stoch_d or 0.0

    _, _, buy_count, sell_count = calculate_td_sequential(df)
    td9 = encode_td9(summarize_td_sequential(buy_count, sell_count))

    zig_lines = calculate_zig_zag(df)
    if zig_lines:
        _, p1, _, p2 = zig_lines[-1]
        zig = encode_zig("up" if p2 > p1 else "down")
    else:
        zig = 0

    return pd.DataFrame(
        {
            f"rsi_{suffix}": [rsi_val],
            f"macd_{suffix}": [macd_val],
            f"signal_{suffix}": [signal_val],
            f"ema10_dev_{suffix}": [ema10_dev],
            f"ema10_change_{suffix}": [ema10_change],
            f"price_above_ema10_{suffix}": [price_above],
            f"rsi_change_{suffix}": [rsi_change],
            f"macd_change_{suffix}": [macd_change],
            f"td9_{suffix}": [td9],
            f"zig_{suffix}": [zig],
            f"adx_{suffix}": [adx_val],
            f"stoch_k_{suffix}": [stoch_k],
            f"stoch_d_{suffix}": [stoch_d],
        },
        index=[current_index[-1]],
    )


def _augment_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the latest feature frame with 1h/4h/1d indicators attached."""

    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df = df.sort_index()

    hourly = df.copy()
    four_hour = _resample_ohlc(hourly, "4h")
    daily = _resample_ohlc(hourly, "1d")

    # Bollinger inputs for golden price features
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(hourly)
    hourly["bb_upper_1h"] = bb_upper
    hourly["bb_lower_1h"] = bb_lower
    hourly["price_1h"] = hourly["close"]

    def _merge_features(base: pd.DataFrame, tf: pd.DataFrame, suffix: str) -> pd.DataFrame:
        feats = _compute_timeframe_features(tf, suffix)
        feats = feats.reindex(base.index, method="ffill").fillna(0)
        price_col = f"price_{suffix}"
        if tf.empty:
            base[price_col] = base["close"]
        else:
            base[price_col] = tf["close"].reindex(base.index, method="ffill").bfill()
        return pd.concat([base, feats], axis=1)

    augmented = hourly.copy()
    augmented = _merge_features(augmented, hourly, "1h")
    augmented = _merge_features(augmented, four_hour, "4h")
    augmented = _merge_features(augmented, daily, "1d")

    # Candlestick patterns per timeframe, forward-filled to hourly index
    pattern_1h = fast_candlestick_patterns(hourly[["open", "high", "low", "close"]], suffix="_1h")
    pattern_4h = fast_candlestick_patterns(four_hour[["open", "high", "low", "close"]], suffix="_4h")
    pattern_1d = fast_candlestick_patterns(daily[["open", "high", "low", "close"]], suffix="_1d")

    for frame in (pattern_1h, pattern_4h, pattern_1d):
        frame.index = frame.index.tz_convert(hourly.index.tz) if hasattr(frame.index, "tz") else frame.index

    pattern_4h = pattern_4h.reindex(hourly.index, method="ffill").fillna(0)
    pattern_1d = pattern_1d.reindex(hourly.index, method="ffill").fillna(0)

    augmented = pd.concat([augmented, pattern_1h, pattern_4h, pattern_1d], axis=1)
    return augmented


def _finalise_feature_frame(df: pd.DataFrame, ticker: str | None = None, start=None, end=None) -> pd.DataFrame:
    """Fill missing feature columns with training-time defaults."""

    df = df.copy()
    df["ticker"] = ticker or ""

    if "timestamp" not in df.columns:
        if df.index.name == "timestamp" or isinstance(df.index, pd.DatetimeIndex):
            df["timestamp"] = df.index

    defaults = default_feature_values(df.columns)
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df
