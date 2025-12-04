"""Shared feature engineering helpers for offline and live pipelines."""
from __future__ import annotations

import ast
import math
import re
import logging
from typing import Iterable, Sequence, Tuple

import pandas as pd
import numpy as np

from indicators import parse_fibonacci
from self_learn import FEATURE_NAMES

logger = logging.getLogger(__name__)

# Number of Fibonacci extension/retracement levels we persist per timeframe
FIB_LEVEL_COUNT = 6
# Live trading expects a 1.272 extension when fewer than six levels are returned
FIB_FALLBACK_LEVEL = 1.272
# All timeframes supported by the historical generator and live workflow
TIMEFRAME_SUFFIXES: Tuple[str, ...] = ("1h", "4h", "1d")
# Indicator columns that expose trailing deltas in FEATURE_NAMES
DELTA_BASE_COLUMNS: Tuple[str, ...] = ("rsi", "macd", "ema10")

# Feature columns that should fall back to domain-specific defaults instead of zero
SPECIAL_NUMERIC_DEFAULTS: dict[str, float] = {
    # Neutral S&P 500 breadth reading used throughout the codebase
    "sp500_above_20d": 50.0,
}


def _is_finite(value: object) -> bool:
    """Return ``True`` when ``value`` can be converted to a finite float."""
    try:
        if value is None:
            return False
        val = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(val) and math.isfinite(val)


def encode_td9(summary: object) -> int:
    """Map TD Sequential summaries to signed setup counts."""
    if not isinstance(summary, str):
        return 0

    text = summary.strip()
    if not text:
        return 0

    tokens = text.split()
    if len(tokens) == 2 and tokens[1].isdigit():
        direction, count = tokens[0].lower(), int(tokens[1])
        if direction == "buy":
            return count
        if direction == "sell":
            return -count
        return 0

    matches = re.findall(r"(buy|sell)\s*=?\s*(\d+)", text, flags=re.IGNORECASE)
    for direction, count_str in matches:
        count = int(count_str)
        if direction.lower() == "buy" and count > 0:
            return count
        if direction.lower() == "sell" and count > 0:
            return -count
    return 0


def encode_zig(trend: object) -> int:
    """Encode ZigZag trend labels as -1/0/1."""
    if not isinstance(trend, str):
        return 0
    label = trend.strip().lower()
    if label == "up":
        return 1
    if label == "down":
        return -1
    return 0


def encode_vol_cat(category: object) -> int:
    """Encode volume categories using the live trading mapping."""
    if not isinstance(category, str):
        return 0
    mapping = {
        "high bull volume": 2,
        "high bear volume": -2,
        "low bull volume": 1,
        "low bear volume": -1,
        "neutral volume": 0,
    }
    return mapping.get(category.strip().lower(), 0)


def ensure_numeric_bool(series: pd.Series) -> pd.Series:
    """Cast a boolean-like series to integers without raising on nulls."""
    if series.empty:
        return series
    return series.fillna(0).astype(int)


def count_fib_timezones(value: object) -> int:
    """Return the number of Fibonacci time zones encoded in ``value``."""
    if value is None:
        return 0
    if isinstance(value, float) and math.isnan(value):
        return 0
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return 0
        parts = [part for part in cleaned.replace("|", ",").split(",") if part.strip()]
        return len(parts)
    if isinstance(value, Iterable):
        return sum(1 for part in value if str(part).strip())
    return 0


def derive_fibonacci_features(summary: object, price: object) -> Tuple[list[float], float]:
    """Return ``(levels, zone_delta)`` derived from the serialized summary."""
    price_value = float(price) if _is_finite(price) else 0.0
    raw_levels = parse_fibonacci(summary, price_value)
    levels = [float(level) if _is_finite(level) else 0.0 for level in raw_levels]
    if len(levels) < FIB_LEVEL_COUNT and price_value:
        levels = levels + [price_value * FIB_FALLBACK_LEVEL]
    if len(levels) < FIB_LEVEL_COUNT:
        levels = levels + [0.0] * (FIB_LEVEL_COUNT - len(levels))
    else:
        levels = levels[:FIB_LEVEL_COUNT]

    fib_prices = [level for level in levels if _is_finite(level)]
    if fib_prices and price_value:
        nearest = min(fib_prices, key=lambda fib_price: abs(price_value - fib_price))
        zone_delta = abs(price_value - nearest) / price_value
    else:
        zone_delta = 0.0
    return levels, zone_delta


def add_golden_price_features(
    df: pd.DataFrame, group_key: str | None = "ticker"
) -> pd.DataFrame:
    """Return ``df`` with hourly return/z-score/Bollinger-position features.

    The helper mirrors the data-prep logic used during training so live trading
    and backtests feed the model the same price-derived signals. It is
    intentionally tolerant: if required inputs are missing the original frame is
    returned unchanged.
    """

    required_columns = {"price_1h", "bb_upper_1h", "bb_lower_1h"}
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        logger.warning(
            "Skipping golden price feature generation; missing required columns: %s",
            missing_required,
        )
        return df

    def _process_group(group: pd.DataFrame) -> pd.DataFrame:
        local = group.copy()

        if "timestamp" in local.columns:
            local["timestamp"] = pd.to_datetime(
                local["timestamp"], utc=True, errors="coerce"
            )
            local = local.dropna(subset=["timestamp"]).sort_values("timestamp")
            local = local.set_index("timestamp")
        elif isinstance(local.index, pd.DatetimeIndex):
            local = local.sort_index()
        else:
            logger.warning(
                "Unable to compute golden price features without a timestamp column or index."
            )
            return group

        local = local.asfreq("h")

        if group_key and group_key in local.columns:
            local[group_key] = group[group_key].iloc[0]

        local = local.reset_index().rename(columns={"index": "timestamp"})

        local["ret_1h"] = np.log(local["price_1h"] / local["price_1h"].shift(1))
        local["ret_4h"] = np.log(local["price_1h"] / local["price_1h"].shift(4))
        local["ret_24h"] = np.log(local["price_1h"] / local["price_1h"].shift(24))

        rolling_mean = local["price_1h"].rolling(window=120, min_periods=60).mean()
        rolling_std = local["price_1h"].rolling(window=120, min_periods=60).std()
        local["price_z_120h"] = (local["price_1h"] - rolling_mean) / rolling_std

        local["bb_position_1h"] = (
            local["price_1h"] - local["bb_lower_1h"]
        ) / (local["bb_upper_1h"] - local["bb_lower_1h"] + 1e-8)

        local["bb_position_1h"] = local["bb_position_1h"].clip(0, 1)
        local["price_z_120h"] = local["price_z_120h"].clip(-5, 5)

        fill_cols = ["ret_1h", "ret_4h", "ret_24h", "price_z_120h", "bb_position_1h"]
        local[fill_cols] = local[fill_cols].ffill().bfill()

        return local

    if group_key and group_key in df.columns:
        processed = df.groupby(group_key, group_keys=False).apply(_process_group)
    else:
        processed = _process_group(df)

    logger.info("Added golden price features (returns, z-score, Bollinger position).")
    return processed


def extract_pivot_prices(value: object) -> list[float]:
    """Return numeric pivot prices from a serialized pivot point mapping."""
    if isinstance(value, dict):
        items = value.values()
    elif isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
        if isinstance(parsed, dict):
            items = parsed.values()
        else:
            return []
    else:
        return []

    prices: list[float] = []
    for item in items:
        if _is_finite(item):
            prices.append(float(item))
    return prices


def compute_indicator_delta(series: pd.Series | pd.DataFrame) -> pd.Series:
    """Return the first difference for an indicator series with NaNs handled."""
    if isinstance(series, pd.DataFrame):
        # Defensive guard — callers should pass a Series, but pandas can occasionally
        # hand back a single-column DataFrame depending on how the column was
        # selected. ``squeeze`` keeps compatibility while ensuring a Series result.
        if series.shape[1] == 1:
            series = series.iloc[:, 0]
        else:
            raise ValueError("compute_indicator_delta expects a single column input")

    if series.empty:
        return series
    return series.diff().fillna(0.0)


def default_feature_values(feature_names: Sequence[str]) -> dict[str, float]:
    """Return default values for ``FEATURE_NAMES`` aligned to training schema."""
    defaults: dict[str, float] = {}
    bool_prefixes = (
        "price_above_",
        "high_vol_",
        "vol_spike_",
        "td9_",
        "zig_",
        "vol_cat_",
        "fib_time_count_",
        "pattern_",
    )

    for name in feature_names:
        if name.startswith(bool_prefixes):
            defaults[name] = 0
        elif name in SPECIAL_NUMERIC_DEFAULTS:
            defaults[name] = SPECIAL_NUMERIC_DEFAULTS[name]
        else:
            defaults[name] = 0.0
    return defaults


def sanitize_feature_row(
    feature_row: pd.Series, feature_names: Sequence[str]
) -> pd.Series:
    """
    铁血三连防御：任何进入模型的特征都不允许出现 NaN/inf。

    专为 PPO 模型设计，强化防御：
    1) 强制转换为 float，非有限值一律归零
    2) 全量扫描 NaN/inf，记录日志并替换
    3) PPO 最敏感特征使用安全默认值，避免“全零”毒瘤
    """

    sanitized = feature_row.reindex(feature_names).copy()

    # 第一步：强制转为 float，NaN/inf 直接变 0
    sanitized = sanitized.astype(float, errors="ignore")

    # 第二步：替换所有非有限值（NaN, inf, -inf）
    mask_bad = ~np.isfinite(sanitized)
    if mask_bad.any():
        bad_cols = sanitized[mask_bad].index.tolist()
        logger.warning("PPO输入含 NaN/inf，已强制替换为0: %s", bad_cols)
        sanitized[mask_bad] = 0.0

    # 第三步：对 PPO 最敏感的几个特征做特殊默认值（而不是0）
    ppo_critical_defaults = {
        "rsi_1h": 50.0,
        "macd_1h": 0.0,
        "macd_signal_1h": 0.0,
        "bb_position_1h": 0.5,
        "price_z_120h": 0.0,
        "atr_1h": 1.0,  # ATR=0 会导致波动率判断爆炸
        "volume_1h": 100000,  # 避免除零
        "sp500_above_20d": 50.0,
    }
    for col, default in ppo_critical_defaults.items():
        if col in sanitized.index:
            if not np.isfinite(sanitized[col]) or sanitized[col] == 0:
                sanitized[col] = default

    return sanitized.astype(np.float32)  # PPO 模型只吃 float32


# ================================
# SEQUENCE CREATION FOR LSTM (FIXED & ROBUST)
# ================================

def _ensure_datetime_index(df: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Convert ndarray → DataFrame with fake timestamps if needed."""
    if isinstance(df, np.ndarray):
        # Live trading path — create fake DataFrame
        if df.ndim != 2:
            raise ValueError("Array must be 2D (rows, features)")
        df = pd.DataFrame(df, columns=FEATURE_NAMES)
        fake_dates = pd.date_range("2020-01-01", periods=len(df), freq="1H", tz="UTC")
        df["timestamp"] = fake_dates
        df = df.set_index("timestamp")
    else:
        df = df.copy()
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")

        timestamps = pd.to_datetime(df['timestamp'])
        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize("UTC")
        else:
            timestamps = timestamps.dt.tz_convert("UTC")

        df['timestamp'] = timestamps
        df = df.set_index('timestamp').sort_index()
    return df


def _get_mapped_features(df: pd.DataFrame, base_features: list[str], suffix: str) -> list[str]:
    """
    Map base 1h features to target timeframe suffix (e.g., rsi_1h → rsi_4h).
    Returns list in same order as base_features.
    """
    mapped = []
    for feat in base_features:
        candidate = feat.replace("_1h", f"_{suffix}") if "_1h" in feat else f"{feat}_{suffix}"
        if candidate in df.columns:
            mapped.append(candidate)
        else:
            mapped.append(feat)  # fallback to 1h name (will be filled with 0 later)
    return mapped


def create_sequence(
    df: pd.DataFrame,
    idx: int,
    seq_len: int = 64,
    timeframes: Tuple[str, ...] = ("1h", "4h", "1d")
) -> np.ndarray:
    """
    Create multi-timeframe sequence for LSTM/Transformer.
    
    Args:
        df: DataFrame with 'timestamp' and multi-timeframe features
        idx: Index in 1H data to center the sequence
        seq_len: Length of output sequence
        timeframes: Which timeframe features to include
    
    Returns:
        (seq_len, total_features) array of type float32
    """
    from self_learn import FEATURE_NAMES_1H  # Base feature list (1h)

    df = _ensure_datetime_index(df)
    if idx >= len(df):
        raise IndexError(f"idx {idx} out of bounds for df of length {len(df)}")

    # Extract 1h window
    start_idx = max(0, idx - seq_len + 1)
    end_idx = idx + 1
    seq_1h_raw = df.iloc[start_idx:end_idx]

    sequences = []
    base_features = FEATURE_NAMES_1H

    for tf in timeframes:
        if tf == "1h":
            rule = "1h"
            feature_cols = [c for c in base_features if c in seq_1h_raw.columns]
            df_tf = seq_1h_raw
        else:
            rule = "4h" if tf == "4h" else "1D"
            # Resample 1h → target timeframe
            df_resampled = seq_1h_raw.resample(rule).last()
            df_filled = df_resampled.asfreq(rule).ffill()  # Critical: preserve grid
            df_tf = df_filled.tail(seq_len)

            # Map features: rsi_1h → rsi_4h, etc.
            feature_cols = _get_mapped_features(df_tf, base_features, tf)

        # Reindex to exact base feature order, fill missing with 0.0
        seq_df = df_tf.reindex(columns=feature_cols)
        seq_aligned = seq_df.reindex(columns=base_features, fill_value=0.0)

        seq = seq_aligned.values.astype(np.float32)

        # Final padding: repeat last row if too short
        if len(seq) < seq_len:
            pad_width = seq_len - len(seq)
            last_row = seq[-1:] if len(seq) > 0 else np.zeros((1, len(base_features)), dtype=np.float32)
            pad = np.repeat(last_row, pad_width, axis=0)
            seq = np.vstack([seq, pad])
        else:
            seq = seq[-seq_len:]

        sequences.append(seq)

    # Stack all timeframes: (seq_len, sum_features_across_timeframes)
    return np.concatenate(sequences, axis=1)