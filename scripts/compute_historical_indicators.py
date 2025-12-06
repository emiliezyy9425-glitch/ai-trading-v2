"""Utilities for computing historical technical indicators.

This module processes locally cached raw price data – downloaded via
``scripts.download_historical_prices`` which pulls from IBKR – and derives the
feature set required by the training pipeline.
"""

import os
import csv
import logging
from pathlib import Path
import sys
from typing import Optional, Tuple
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from csv_utils import save_dataframe_with_timestamp_validation
from feature_engineering import (
    DELTA_BASE_COLUMNS,
    FIB_LEVEL_COUNT,
    compute_indicator_delta,
    count_fib_timezones,
    default_feature_values,
    derive_fibonacci_features,
    encode_td9,
    encode_vol_cat,
    encode_zig,
    ensure_numeric_bool,
    extract_pivot_prices,
)
from indicators import (
    calculate_td_sequential,
    calculate_tds_trend,
    detect_pivots,
    calculate_fib_levels_from_pivots,
    calculate_fib_time_zones,
    calculate_pivot_points,
    calculate_zig_zag,
    calculate_high_volatility,
    calculate_volume_spike,
    calculate_volume_weighted_category,
    summarize_td_sequential,
    calculate_adx,
    calculate_obv,
    calculate_stochastic_oscillator,
    calculate_level_weight,
    fast_candlestick_patterns,
    DEFAULT_CANDLESTICK_PATTERN_NAMES,
    DEFAULT_CANDLESTICK_PATTERN_COLUMNS,
    DEFAULT_CANDLESTICK_PATTERN_CODES,
)
from self_learn import FEATURE_NAMES
from training_lookback import (
    get_training_lookback_days,
    get_training_lookback_years,
)

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'compute_historical_indicators.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

LOOKBACK_YEARS = get_training_lookback_years()
# Fetch up to ~10 years of data when resampling cached IBKR data
LOOKBACK_DAYS = get_training_lookback_days()
# Maximum number of bars required by any indicator (e.g., MACD slow period)
INDICATOR_LOOKBACK_BARS = 26

SELECTED_CANDLESTICK_FEATURES = list(DEFAULT_CANDLESTICK_PATTERN_COLUMNS)

PATTERN_TIMEFRAME_SPECS = [
    ("1 hour", "1h"),
    ("4 hours", "4h"),
    ("1 day", "1d"),
]
TIMEFRAME_DISPLAY_TO_SUFFIX = dict(PATTERN_TIMEFRAME_SPECS)
TIMEFRAME_ORDER = [display for display, _ in PATTERN_TIMEFRAME_SPECS]
TIMEFRAME_SUFFIXES = tuple(suffix for _, suffix in PATTERN_TIMEFRAME_SPECS)
_BASE_PATTERN_SPECS: list[tuple[str, str, str]] = [
    (name, column, code)
    for name, column, code in zip(
        DEFAULT_CANDLESTICK_PATTERN_NAMES,
        DEFAULT_CANDLESTICK_PATTERN_COLUMNS,
        DEFAULT_CANDLESTICK_PATTERN_CODES,
    )
]
PATTERN_FEATURE_SPECS = [
    (
        timeframe,
        display_name,
        feature if suffix == "1h" else f"{feature}_{suffix}",
    )
    for timeframe, suffix in PATTERN_TIMEFRAME_SPECS
    for display_name, feature, _ in _BASE_PATTERN_SPECS
]

FEATURE_DEFAULTS = default_feature_values(FEATURE_NAMES)
PRICE_COLUMNS = [f"price_{suffix}" for suffix in TIMEFRAME_SUFFIXES]
OHLC_FIELDS = ("open", "high", "low", "close")
TIMEFRAME_OHLC_COLUMNS = [
    f"{field}_{suffix}" for suffix in TIMEFRAME_SUFFIXES for field in OHLC_FIELDS
]
BOOL_FEATURE_PREFIXES = ("price_above_", "high_vol_", "vol_spike_", "pattern_")
REQUIRED_PRICE_COLUMNS = {"open", "high", "low", "close", "volume"}


def _trim_lookback_window(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` filtered to the configured lookback horizon."""

    if df.empty:
        return df

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)
    mask = (df.index >= start) & (df.index <= end)
    return df.loc[mask]


def _is_bool_feature(name: str) -> bool:
    return name.startswith(BOOL_FEATURE_PREFIXES)


def _empty_pattern_frame(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(0, index=index, columns=SELECTED_CANDLESTICK_FEATURES)


def _compute_selected_candlestick_patterns(
    df: pd.DataFrame, timeframe_suffix: str
) -> pd.DataFrame:
    """Return the selected candlestick pattern columns for ``df``."""

    if df is None or df.empty:
        return _empty_pattern_frame(df.index if hasattr(df, "index") else pd.Index([]))

    pattern_frame: pd.DataFrame | None = None
    try:
        from indicators import get_candlestick_patterns

        patterns = get_candlestick_patterns(df, suffix=f"_{timeframe_suffix}")
        if isinstance(patterns, pd.DataFrame):
            pattern_frame = patterns.rename(
                columns=lambda x: x.replace(f"_{timeframe_suffix}", "")
            )
    except TypeError:
        # ``get_candlestick_patterns`` in some revisions does not accept ``suffix``;
        # fall back to the fast vectorised implementation below.
        pattern_frame = None
    except Exception as exc:
        logger.warning(
            "Candlestick patterns failed for %s via helper: %s", timeframe_suffix, exc
        )
        pattern_frame = None

    if pattern_frame is None:
        try:
            suffix = f"_{timeframe_suffix}" if timeframe_suffix else ""
            pattern_frame = fast_candlestick_patterns(df, suffix=suffix).rename(
                columns=lambda x: x.replace(suffix, "")
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to compute candlestick patterns: %s", exc)
            return _empty_pattern_frame(df.index)

    if pattern_frame.empty:
        return _empty_pattern_frame(df.index)

    subset = pattern_frame.reindex(columns=SELECTED_CANDLESTICK_FEATURES, fill_value=0)
    return subset.astype(int)


def _latest_td_value(sequence, explicit: int | float | None = None) -> int:
    """Return the most recent TD Sequential count from ``sequence``."""

    if explicit is not None:
        try:
            return int(explicit)
        except (TypeError, ValueError):
            pass

    if isinstance(sequence, pd.Series):
        if sequence.empty:
            return 0
        try:
            return int(sequence.iloc[-1])
        except (TypeError, ValueError):
            return 0

    if isinstance(sequence, (list, tuple, np.ndarray)):
        if len(sequence) == 0:  # type: ignore[arg-type]
            return 0
        try:
            return int(sequence[-1])  # type: ignore[index]
        except (TypeError, ValueError):
            return 0

    if sequence is None:
        return 0

    try:
        return int(sequence)
    except (TypeError, ValueError):
        return 0


def _td_sequential_counts(result) -> tuple[int, int]:
    """Return ``(buy_count, sell_count)`` from ``calculate_td_sequential`` output."""

    buy_series = sell_series = None
    latest_buy = latest_sell = None

    if isinstance(result, tuple):
        if len(result) >= 4:
            buy_series, sell_series, latest_buy, latest_sell = result[:4]
        elif len(result) >= 2:
            buy_series, sell_series = result[:2]
        elif len(result) == 1:
            buy_series = result[0]
    else:
        buy_series = result

    buy_count = _latest_td_value(buy_series, latest_buy)
    sell_count = _latest_td_value(sell_series, latest_sell)
    return buy_count, sell_count


def _resolve_zig_zag_trend(zig_result, df_slice: pd.DataFrame) -> str:
    """Return an "Up"/"Down"/"N/A" label from ``calculate_zig_zag`` output."""

    if isinstance(zig_result, pd.Series):
        pivots = zig_result[zig_result == 1].index
        if len(pivots) >= 2 and "close" in df_slice:
            last_price = df_slice.loc[pivots[-1], "close"]
            prev_price = df_slice.loc[pivots[-2], "close"]
            if last_price > prev_price:
                return "Up"
            if last_price < prev_price:
                return "Down"
        return "N/A"

    if isinstance(zig_result, (list, tuple)):
        iterable = zig_result if isinstance(zig_result, list) else list(zig_result)
        for segment in reversed(iterable):
            if not isinstance(segment, (list, tuple)) or len(segment) < 4:
                continue
            _, price_start, _, price_end = segment[:4]
            try:
                start = float(price_start)
                end = float(price_end)
            except (TypeError, ValueError):
                continue
            if end > start:
                return "Up"
            if end < start:
                return "Down"
        return "N/A"

    return "N/A"


def _compute_core_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI, MACD, Bollinger Bands, Stochastic, and ATR series."""

    result = pd.DataFrame(index=df.index)

    # RSI
    result["rsi"] = rsi_series(df["close"])

    # MACD
    macd_line, signal = macd_series(df["close"])
    result["macd"] = macd_line
    result["signal"] = signal

    # Bollinger Bands
    upper, mid, lower = bollinger_bands(df["close"])
    result["bb_upper"] = upper
    result["bb_mid"] = mid
    result["bb_lower"] = lower

    # Stochastic Oscillator
    try:
        k, d = calculate_stochastic_oscillator(df["high"], df["low"], df["close"])
    except Exception:  # pragma: no cover - defensive fallback
        k, d = (pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index))
    if not isinstance(k, pd.Series):
        k = pd.Series(k, index=df.index)
    if not isinstance(d, pd.Series):
        d = pd.Series(d, index=df.index)
    result["stoch_k"] = k.reindex(df.index)
    result["stoch_d"] = d.reindex(df.index)

    # ATR
    result["atr"] = atr_series(df["high"], df["low"], df["close"])

    return result


def _encode_timeframe_features(features: pd.DataFrame, suffix: str) -> None:
    """Add delta, momentum and derived encodings for a timeframe (in-place)."""
    td9_col = f"td9_summary_{suffix}"
    zig_col = f"zig_zag_trend_{suffix}"
    vol_col = f"vol_category_{suffix}"
    tz_col = f"fib_time_zones_{suffix}"
    fib_summary_col = f"fib_summary_{suffix}"
    price_col = f"price_{suffix}"

    if td9_col in features:
        features[f"td9_{suffix}"] = features[td9_col].apply(encode_td9).fillna(0).astype(int)
    else:
        features[f"td9_{suffix}"] = 0

    if zig_col in features:
        features[f"zig_{suffix}"] = features[zig_col].apply(encode_zig).fillna(0).astype(int)
    else:
        features[f"zig_{suffix}"] = 0

    if vol_col in features:
        features[f"vol_cat_{suffix}"] = features[vol_col].apply(encode_vol_cat).fillna(0).astype(int)
    else:
        features[f"vol_cat_{suffix}"] = 0

    if tz_col in features:
        features[f"fib_time_count_{suffix}"] = (
            features[tz_col].apply(count_fib_timezones).fillna(0).astype(int)
        )
    else:
        features[f"fib_time_count_{suffix}"] = 0

    if fib_summary_col in features and price_col in features:
        fib_levels: list[list[float]] = []
        fib_zone_deltas: list[float] = []
        for summary, price in zip(features[fib_summary_col], features[price_col]):
            levels, zone_delta = derive_fibonacci_features(summary, price)
            fib_levels.append(levels)
            fib_zone_deltas.append(zone_delta)
        if fib_levels:
            for idx in range(FIB_LEVEL_COUNT):
                column_name = f"fib_level{idx + 1}_{suffix}"
                features[column_name] = pd.Series(
                    [levels[idx] for levels in fib_levels], index=features.index
                )
        else:
            for idx in range(FIB_LEVEL_COUNT):
                features[f"fib_level{idx + 1}_{suffix}"] = 0.0
        if fib_zone_deltas:
            features[f"fib_zone_delta_{suffix}"] = pd.Series(
                fib_zone_deltas, index=features.index
            )
        else:
            features[f"fib_zone_delta_{suffix}"] = 0.0
    else:
        for idx in range(FIB_LEVEL_COUNT):
            features[f"fib_level{idx + 1}_{suffix}"] = 0.0
        features[f"fib_zone_delta_{suffix}"] = 0.0

    for base in DELTA_BASE_COLUMNS:
        column = f"{base}_{suffix}"
        delta_column = f"{base}_change_{suffix}"
        if column in features:
            features[delta_column] = compute_indicator_delta(features[column])
        else:
            features[delta_column] = 0.0

    # ------------------------------------------------------------------
    # Additional delta/momentum encodings appended to original signals
    # ------------------------------------------------------------------
    rsi_col = f"rsi_{suffix}"
    rsi_series = _safe_series(rsi_col)
    if rsi_series is not None:
        features[f"delta_rsi_{suffix}"] = compute_indicator_delta(rsi_series)
        features[f"rsi_momentum_{suffix}"] = rsi_series.diff(3)

    macd_col = f"macd_{suffix}"
    signal_col = f"signal_{suffix}"
    macd_series = _safe_series(macd_col)
    signal_series = _safe_series(signal_col)

    if macd_series is not None:
        features[f"delta_macd_{suffix}"] = compute_indicator_delta(macd_series)

    if macd_series is not None and signal_series is not None:
        hist = macd_series - signal_series
        features[f"macd_hist_{suffix}"] = hist
        features[f"delta_macd_hist_{suffix}"] = compute_indicator_delta(hist)

    bb_pos_col = f"bb_position_{suffix}"
    bb_width_col = f"bb_width_{suffix}"

    if bb_pos_col in features.columns:
        features[f"delta_bb_position_{suffix}"] = compute_indicator_delta(features[bb_pos_col])

    if bb_width_col in features.columns:
        features[f"delta_bb_width_{suffix}"] = compute_indicator_delta(features[bb_width_col])

    adx_col = f"adx_{suffix}"
    if adx_col in features.columns:
        features[f"delta_adx_{suffix}"] = compute_indicator_delta(features[adx_col])

    stoch_k = f"stoch_k_{suffix}"
    stoch_d = f"stoch_d_{suffix}"
    if stoch_k in features.columns:
        features[f"delta_stoch_k_{suffix}"] = compute_indicator_delta(features[stoch_k])
    if stoch_d in features.columns:
        features[f"delta_stoch_d_{suffix}"] = compute_indicator_delta(features[stoch_d])

    obv_col = f"obv_{suffix}"
    if obv_col in features.columns:
        features[f"delta_obv_{suffix}"] = compute_indicator_delta(features[obv_col])


def _apply_timeframe_encodings(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(index=features.index)

    features = features.copy()
    for suffix in TIMEFRAME_SUFFIXES:
        _encode_timeframe_features(features, suffix)

        for column in (
            f"price_above_ema10_{suffix}",
            f"high_vol_{suffix}",
            f"vol_spike_{suffix}",
        ):
            if column in features:
                features[column] = ensure_numeric_bool(features[column])

    pattern_columns = [col for col in features.columns if col.startswith("pattern_")]
    for column in pattern_columns:
        if column.endswith("_1h"):
            base_name = column[:-3]
            features[base_name] = features[column]

    return features


def _compute_level_weights(features: pd.DataFrame) -> pd.Series:
    def _row_weight(row: pd.Series) -> float:
        price_candidates = [
            row.get("price_1h"),
            row.get("price_4h"),
            row.get("price_1d"),
            row.get("price"),
        ]
        price = 0.0
        for candidate in price_candidates:
            try:
                if pd.notna(candidate) and float(candidate) != 0.0:
                    price = float(candidate)
                    break
            except (TypeError, ValueError):
                continue

        fib_prices: list[float] = []
        pivot_prices: list[float] = []
        for suffix in TIMEFRAME_SUFFIXES:
            price_key = f"price_{suffix}"
            price_val = row.get(price_key)
            try:
                price_float = float(price_val)
            except (TypeError, ValueError):
                continue
            if not pd.notna(price_float) or price_float == 0.0:
                continue

            for idx in range(1, FIB_LEVEL_COUNT + 1):
                level_val = row.get(f"fib_level{idx}_{suffix}")
                try:
                    level_float = float(level_val)
                except (TypeError, ValueError):
                    continue
                if not pd.notna(level_float) or level_float == 0.0:
                    continue
                fib_prices.append(level_float)

            pivot_prices.extend(extract_pivot_prices(row.get(f"pivot_points_{suffix}")))

        try:
            return float(calculate_level_weight(price, fib_prices, pivot_prices))
        except Exception:  # pragma: no cover - defensive only
            return 0.0

    if features.empty:
        return pd.Series(dtype=float)

    return features.apply(_row_weight, axis=1)


def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "timestamp" not in df:
        df["timestamp"] = pd.NaT
    if "ticker" not in df:
        df["ticker"] = ""

    for column in [*TIMEFRAME_OHLC_COLUMNS, *PRICE_COLUMNS]:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = 0.0

    missing_defaults = {
        feature: FEATURE_DEFAULTS[feature]
        for feature in FEATURE_NAMES
        if feature not in df.columns
    }

    if missing_defaults:
        defaults_frame = pd.DataFrame(missing_defaults, index=df.index)
        df = pd.concat([df, defaults_frame], axis=1)

    bool_features = [
        feature for feature in FEATURE_NAMES if _is_bool_feature(feature) and feature in df.columns
    ]
    for feature in bool_features:
        df[feature] = ensure_numeric_bool(df[feature])
        df[feature] = df[feature].fillna(int(FEATURE_DEFAULTS[feature]))

    numeric_features = [
        feature for feature in FEATURE_NAMES if feature in df.columns and not _is_bool_feature(feature)
    ]
    if numeric_features:
        df[numeric_features] = df[numeric_features].apply(pd.to_numeric, errors="coerce")
        fill_values = {feature: FEATURE_DEFAULTS[feature] for feature in numeric_features}
        df[numeric_features] = df[numeric_features].fillna(fill_values)

    ordered_columns = ["timestamp", "ticker", *TIMEFRAME_OHLC_COLUMNS, *PRICE_COLUMNS, *FEATURE_NAMES]
    for column in ordered_columns:
        if column not in df.columns:
            df[column] = FEATURE_DEFAULTS.get(column, 0.0)

    return df[ordered_columns]


def rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd_series(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def bollinger_bands(close: pd.Series, window: int = 20, num_std: int = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def atr_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Return ATR series, or NaNs if source data is insufficient."""
    if high is None or low is None or close is None:
        return pd.Series([np.nan] * (len(high) if high is not None else 0), index=high.index if high is not None else None)
    prices = pd.concat([high, low, close], axis=1)
    if prices.isnull().any().any() or len(prices) < period:
        return pd.Series([np.nan] * len(prices), index=prices.index)
    high_low = high - low
    high_close_prev = (high - close.shift()).abs()
    low_close_prev = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def _ensure_datetime_index(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """Return ``df`` with a timezone-aware ``DatetimeIndex``.

    Some call-sites – notably the historical dataset generator – may provide
    raw frames indexed by integers. Pandas resampling requires a
    ``DatetimeIndex`` so normalise inputs defensively before indicator
    calculation.
    """

    if df is None:
        return pd.DataFrame()

    working = df.copy()

    if isinstance(working.index, pd.DatetimeIndex):
        if working.index.tz is None:
            working.index = working.index.tz_localize(timezone.utc)
        else:
            working.index = working.index.tz_convert(timezone.utc)
        working = working.sort_index()
        working.index.name = working.index.name or "timestamp"
        return working

    if "timestamp" not in working.columns:
        logger.error("%s dataframe is missing 'timestamp' column", context or "Input")
        return pd.DataFrame()

    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    working = working.dropna(subset=["timestamp"])
    if working.empty:
        return pd.DataFrame()

    working = working.sort_values("timestamp").set_index("timestamp")
    working.index = working.index.tz_convert(timezone.utc)
    working.index.name = working.index.name or "timestamp"
    return working


def compute_indicators(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    df = _ensure_datetime_index(df, f"compute_indicators[{timeframe}]")
    if df is None or df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz=timezone.utc, name="timestamp"))

    out = pd.DataFrame(index=df.index)
    out['rsi'] = rsi_series(df['close'])
    macd, signal = macd_series(df['close'])
    out['macd'] = macd
    out['signal'] = signal
    
    # Compute 10-day EMA from daily close prices
    if timeframe == '1d':
        out['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    else:
        daily_close = df['close'].resample('1D').last()
        daily_ema = daily_close.ewm(span=10, adjust=False).mean()
        # Align the higher timeframe EMA to the next session so intraday rows only
        # see fully closed daily candles (prevents inadvertent lookahead bias).
        daily_ema = daily_ema.shift(1)
        out['ema10'] = daily_ema.reindex(df.index, method='ffill')
    
    out['price_above_ema10'] = (df['close'] > out['ema10']).astype(int)
    out['ema10_dev'] = np.where(
        out['ema10'] != 0,
        (df['close'] - out['ema10']) / out['ema10'],
        0.0,
    )
    bb_upper, bb_mid, bb_lower = bollinger_bands(df['close'])
    out['bb_upper'] = bb_upper
    out['bb_lower'] = bb_lower
    out['bb_mid'] = bb_mid
    out['atr'] = atr_series(df['high'], df['low'], df['close'])
    out['volume'] = df['volume']
    out['price'] = df['close']
    
    return out


def compute_advanced_indicators(df: pd.DataFrame, timeframe_suffix: str) -> pd.DataFrame:
    """Compute advanced indicators that previously used placeholders."""
    results = {
        'td9_summary': [],
        'tds_trend': [],
        'tds_signal': [],
        'fib_summary': [],
        'fib_time_zones': [],
        'pivot_points': [],
        'zig_zag_trend': [],
        'high_vol': [],
        'vol_spike': [],
        'vol_category': [],
    }

    for i in range(len(df)):
        slice_df = df.iloc[: i + 1]
        td_result = calculate_td_sequential(slice_df)
        buy_count, sell_count = _td_sequential_counts(td_result)
        results['td9_summary'].append(
            summarize_td_sequential(buy_count, sell_count)
        )

        tds_trend_val, tds_signal_val = calculate_tds_trend(slice_df)
        results['tds_trend'].append(tds_trend_val)
        results['tds_signal'].append(tds_signal_val)

        pivots = detect_pivots(slice_df)
        fib_levels = calculate_fib_levels_from_pivots(pivots)
        results['fib_summary'].append(str(fib_levels) if fib_levels else 'N/A')
        fib_times = calculate_fib_time_zones(pivots, slice_df.index[-1])
        results['fib_time_zones'].append(','.join(map(str, fib_times[:5])) if fib_times else '')
        pivot_pts = calculate_pivot_points(slice_df)
        results['pivot_points'].append(str(pivot_pts) if pivot_pts else 'N/A')

        zig = calculate_zig_zag(slice_df)
        results['zig_zag_trend'].append(_resolve_zig_zag_trend(zig, slice_df))

        results['high_vol'].append(calculate_high_volatility(slice_df))
        results['vol_spike'].append(calculate_volume_spike(slice_df))
        results['vol_category'].append(calculate_volume_weighted_category(slice_df))

    pattern_df = _compute_selected_candlestick_patterns(df, timeframe_suffix)
    for column in SELECTED_CANDLESTICK_FEATURES:
        results[column] = pattern_df[column].astype(int).tolist()

    return pd.DataFrame(results, index=df.index)

def compute_complex_indicators(df: pd.DataFrame, timeframe_suffix: str) -> pd.DataFrame:
    """Compute advanced indicators that previously defaulted to placeholders.

    The calculations mirror the logic used in the live trading script so that
    historical datasets contain realistic values instead of ``'N/A'`` or zeros.
    """

    td9_summary = []
    tds_trend_vals = []
    tds_signal_vals = []
    fib_summary_vals = []
    fib_time_vals = []
    pivot_points_vals = []
    zig_trend_vals = []
    high_vol_vals = []
    vol_spike_vals = []
    vol_category_vals = []
    pattern_df = _compute_selected_candlestick_patterns(df, timeframe_suffix)

    for i in range(len(df)):
        df_slice = df.iloc[: i + 1]

        try:
            td_result = calculate_td_sequential(df_slice)
            buy_count, sell_count = _td_sequential_counts(td_result)
            td9_summary.append(summarize_td_sequential(buy_count, sell_count))
        except Exception:
            td9_summary.append("N/A")

        try:
            trend_val, signal_val = calculate_tds_trend(df_slice)
        except Exception:
            trend_val, signal_val = 0, 0
        tds_trend_vals.append(trend_val if trend_val is not None else 0)
        tds_signal_vals.append(signal_val if signal_val is not None else 0)

        try:
            pivots = detect_pivots(df_slice)
            fib_levels = calculate_fib_levels_from_pivots(pivots)
            fib_summary_vals.append(
                ",".join([f"{k}:{v}" for k, v in fib_levels.items()]) if fib_levels else "N/A"
            )
            fib_time = calculate_fib_time_zones(pivots, df_slice.index[-1])
            fib_time_vals.append(
                "|".join([t.isoformat() for t in fib_time]) if fib_time else ""
            )
            pivot_pts = calculate_pivot_points(df_slice)
            pivot_points_vals.append(
                ",".join([f"{k}:{v}" for k, v in pivot_pts.items()]) if pivot_pts else "N/A"
            )
        except Exception:
            fib_summary_vals.append("N/A")
            fib_time_vals.append("")
            pivot_points_vals.append("N/A")

        try:
            if len(df_slice) >= INDICATOR_LOOKBACK_BARS:
                zig = calculate_zig_zag(df_slice)
                zig_trend_vals.append(_resolve_zig_zag_trend(zig, df_slice))
            else:
                zig_trend_vals.append("N/A")
        except Exception:
            zig_trend_vals.append("N/A")

        try:
            high_vol_vals.append(bool(calculate_high_volatility(df_slice)))
        except Exception:
            high_vol_vals.append(False)

        try:
            vol_spike_vals.append(bool(calculate_volume_spike(df_slice)))
        except Exception:
            vol_spike_vals.append(False)

        try:
            vol_category_vals.append(
                calculate_volume_weighted_category(df_slice) or "Neutral Volume"
            )
        except Exception:
            vol_category_vals.append("Neutral Volume")

    return pd.DataFrame(
        {
            "td9_summary": td9_summary,
            "tds_trend": tds_trend_vals,
            "tds_signal": tds_signal_vals,
            "fib_summary": fib_summary_vals,
            "fib_time_zones": fib_time_vals,
            "pivot_points": pivot_points_vals,
            "zig_zag_trend": zig_trend_vals,
            "high_vol": high_vol_vals,
            "vol_spike": vol_spike_vals,
            "vol_category": vol_category_vals,
            **{col: pattern_df[col].astype(int).tolist() for col in SELECTED_CANDLESTICK_FEATURES},
        },
        index=df.index,
    )



def _prepare_timeframe_features(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Compute indicator features for ``df`` and namespace them by timeframe."""

    if df is None or df.empty:
        return pd.DataFrame()

    suffix = TIMEFRAME_DISPLAY_TO_SUFFIX.get(timeframe)
    if suffix is None:
        logger.warning("Unsupported timeframe %s. Skipping.", timeframe)
        return pd.DataFrame()

    try:
        base_indicators = compute_indicators(df, suffix)
        advanced_indicators = compute_complex_indicators(df, suffix)
    except Exception as exc:  # pragma: no cover - indicator edge cases
        logger.error(
            "Failed to compute indicators for %s on %s: %s",
            getattr(df.index, "name", "series"),
            timeframe,
            exc,
        )
        return pd.DataFrame()

    advanced_indicators = advanced_indicators.drop(
        columns=list(SELECTED_CANDLESTICK_FEATURES), errors="ignore"
    )
    combined = pd.concat([base_indicators, advanced_indicators], axis=1)

    core = _compute_core_indicators(df)
    if not core.empty:
        # Avoid duplicating columns that are already present from compute_indicators
        # (e.g., rsi/macd/bollinger/atr). Keep only the truly missing series such as
        # stochastic values to prevent duplicate column names before suffixing.
        core = core.drop(columns=core.columns.intersection(combined.columns))
        if not core.empty:
            combined = pd.concat([combined, core], axis=1)

    patterns = _compute_selected_candlestick_patterns(df, suffix)
    if not patterns.empty:
        combined = pd.concat([combined, patterns], axis=1)

    if combined.empty:
        return pd.DataFrame()

    def _assign_indicator(column: str, values):
        if isinstance(values, pd.Series):
            combined[column] = values.reindex(combined.index)
        elif values is None:
            combined[column] = np.nan
        else:
            combined[column] = float(values)

    try:
        adx_values = calculate_adx(df["high"], df["low"], df["close"])
        _assign_indicator("adx", adx_values)
    except Exception:
        combined["adx"] = np.nan

    try:
        obv_values = calculate_obv(df["close"], df["volume"])
        _assign_indicator("obv", obv_values)
    except Exception:
        combined["obv"] = np.nan

    try:
        level_weight = calculate_level_weight(combined, timeframe=suffix)
        _assign_indicator("level_weight", level_weight)
    except Exception:
        combined["level_weight"] = np.nan

    combined = combined.reset_index().rename(columns={"index": "timestamp"})
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["timestamp"])
    combined = combined.sort_values("timestamp").drop_duplicates(subset="timestamp")

    rename_map = {
        column: f"{column}_{suffix}"
        for column in combined.columns
        if column not in {"timestamp"}
    }
    combined = combined.rename(columns=rename_map)
    combined["timestamp_ns"] = combined["timestamp"].dt.tz_convert(None).astype("int64")

    return combined


def load_raw_data(input_dir: str, ticker: str) -> Optional[pd.DataFrame]:
    path = os.path.join(input_dir, f"{ticker}.csv")
    if not os.path.exists(path):
        logger.warning(f"Raw data for {ticker} not found at {path}")
        return None
    df = pd.read_csv(path)
    if 'timestamp' not in df.columns:
        logger.error(f"{path} missing 'timestamp' column")
        return None
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp').set_index('timestamp')
    return df[['open', 'high', 'low', 'close', 'volume']]


def resample_df(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample ``df`` using ``rule`` while normalizing the rule string.

    Pandas is deprecating the use of upper-case offset aliases such as
    ``'H'`` for hours. To remain forward compatible, ensure the rule is
    lowercase before passing it to ``DataFrame.resample``.
    """

    rule = rule.lower()
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    return df.resample(rule).agg(agg).dropna()


def _merge_timeframe_features(timeframe_features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not timeframe_features:
        return pd.DataFrame()

    base_timeframe = next((tf for tf in TIMEFRAME_ORDER if tf in timeframe_features), None)
    if base_timeframe is None:
        return pd.DataFrame()

    combined = timeframe_features[base_timeframe].sort_values("timestamp_ns").copy()

    for timeframe in TIMEFRAME_ORDER:
        if timeframe == base_timeframe:
            continue
        tf_features = timeframe_features.get(timeframe)
        if tf_features is None or tf_features.empty:
            continue

        tf_sorted = tf_features.sort_values("timestamp_ns")
        combined = pd.merge_asof(
            combined,
            tf_sorted.drop(columns=["timestamp"]),
            on="timestamp_ns",
            direction="backward",
        )

    combined = combined.sort_values("timestamp_ns").dropna(subset=["timestamp"])
    combined = combined.drop(columns=["timestamp_ns"]).reset_index(drop=True)
    return combined


def compute_for_ticker(raw_dir: Path, ticker: str) -> pd.DataFrame:
    """Fetch data for ``ticker`` and compute indicators for multiple timeframes."""

    raw_df = load_raw_data(str(raw_dir), ticker)
    if raw_df is not None:
        raw_df = _trim_lookback_window(raw_df)
    resampled_cache: dict[str, pd.DataFrame] = {}
    timeframe_features: dict[str, pd.DataFrame] = {}

    for timeframe in TIMEFRAME_ORDER:
        if raw_df is None or raw_df.empty:
            logger.warning("No IBKR data available for %s; skipping timeframe %s.", ticker, timeframe)
            continue

        suffix = TIMEFRAME_DISPLAY_TO_SUFFIX.get(timeframe)
        if suffix is None:
            logger.warning("Unsupported timeframe %s for %s. Skipping.", timeframe, ticker)
            continue

        if timeframe == "1 hour":
            df = raw_df.copy()
        else:
            if suffix not in resampled_cache:
                resampled_cache[suffix] = _trim_lookback_window(resample_df(raw_df, suffix))
            df = resampled_cache.get(suffix)

        if df is None or df.empty:
            logger.warning("No resampled data for %s on %s. Skipping.", ticker, timeframe)
            continue

        logger.info("Using IBKR-derived data for %s on %s timeframe.", ticker, timeframe)

        df = df.sort_index()

        features = _prepare_timeframe_features(df, timeframe)
        if features is None or features.empty:
            logger.warning("Failed to compute features for %s on %s.", ticker, timeframe)
            continue

        timeframe_features[timeframe] = features

    if not timeframe_features:
        return pd.DataFrame()

    combined = _merge_timeframe_features(timeframe_features)
    if combined.empty:
        return pd.DataFrame()
    combined["ticker"] = ticker

    return combined


def _normalise_input_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    ticker_value = ""
    if isinstance(df, pd.DataFrame) and "ticker" in df.columns:
        ticker_series = df["ticker"].dropna()
        if not ticker_series.empty:
            ticker_value = str(ticker_series.iloc[-1])

    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(), ticker_value

    working = df.copy()
    if "timestamp" not in working.columns:
        if isinstance(working.index, pd.DatetimeIndex):
            working = working.reset_index().rename(columns={working.index.name or "index": "timestamp"})
        else:
            logger.error("Input dataframe missing 'timestamp' column")
            return pd.DataFrame(), ticker_value

    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    working = working.dropna(subset=["timestamp"])
    working = working.sort_values("timestamp").drop_duplicates("timestamp")
    if working.empty:
        return pd.DataFrame(), ticker_value

    missing = REQUIRED_PRICE_COLUMNS - set(working.columns)
    if missing:
        logger.error("Input dataframe missing required columns: %s", ", ".join(sorted(missing)))
        return pd.DataFrame(), ticker_value

    working = working.set_index("timestamp")
    return working[list(REQUIRED_PRICE_COLUMNS)], ticker_value


def compute_indicators_for_df(df: pd.DataFrame) -> pd.DataFrame:
    price_frame, ticker_value = _normalise_input_frame(df)
    if price_frame.empty:
        return pd.DataFrame()

    resampled_cache: dict[str, pd.DataFrame] = {}
    timeframe_features: dict[str, pd.DataFrame] = {}

    for timeframe in TIMEFRAME_ORDER:
        if timeframe == "1 hour":
            tf_df = price_frame.copy()
        else:
            suffix = TIMEFRAME_DISPLAY_TO_SUFFIX.get(timeframe)
            if suffix not in resampled_cache:
                resampled_cache[suffix] = resample_df(price_frame, suffix)
            tf_df = resampled_cache.get(suffix)

        if tf_df is None or tf_df.empty:
            continue

        features = _prepare_timeframe_features(tf_df, timeframe)
        if features is None or features.empty:
            continue
        timeframe_features[timeframe] = features

    combined = _merge_timeframe_features(timeframe_features)
    if combined.empty:
        return pd.DataFrame()

    combined = combined.set_index("timestamp")
    encoded = _apply_timeframe_encodings(combined)
    if encoded.empty:
        return pd.DataFrame()

    encoded = encoded.replace([np.inf, -np.inf], np.nan)
    encoded = encoded.dropna(how="all")
    encoded["level_weight"] = _compute_level_weights(encoded.reset_index())
    encoded["ticker"] = ticker_value
    encoded = encoded.reset_index().rename(columns={"index": "timestamp"})
    encoded = encoded.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    def _broadcast_ohlc(source: pd.DataFrame | None, suffix: str, shift_resampled: bool) -> pd.DataFrame:
        columns = {field: f"{field}_{suffix}" for field in OHLC_FIELDS}
        if source is None or source.empty:
            return pd.DataFrame(index=price_frame.index, columns=list(columns.values()))
        working = source[list(OHLC_FIELDS)].rename(columns=columns)
        if shift_resampled:
            working = working.shift(1)
        return working.reindex(price_frame.index).ffill()

    ohlc_frames: list[pd.DataFrame] = []
    ohlc_frames.append(_broadcast_ohlc(price_frame, "1h", shift_resampled=False))
    for suffix in TIMEFRAME_SUFFIXES:
        if suffix == "1h":
            continue
        source = resampled_cache.get(suffix)
        if source is None:
            resampled_cache[suffix] = resample_df(price_frame, suffix)
            source = resampled_cache.get(suffix)
        ohlc_frames.append(_broadcast_ohlc(source, suffix, shift_resampled=True))

    ohlc_combined = pd.concat(ohlc_frames, axis=1)
    ohlc_combined = ohlc_combined.reset_index().rename(columns={"index": "timestamp"})

    merged = pd.merge(encoded, ohlc_combined, on="timestamp", how="left")

    volume_frame = (
        price_frame[["volume"]]
        .rename(columns={"volume": "volume_1h"})
        .reset_index()
        .rename(columns={"index": "timestamp"})
    )
    merged = merged.merge(volume_frame, on="timestamp", how="left")

    for suffix in TIMEFRAME_SUFFIXES:
        close_col = f"close_{suffix}"
        price_col = f"price_{suffix}"
        if close_col in merged:
            merged[price_col] = merged[close_col]

    for suffix in TIMEFRAME_SUFFIXES:
        if suffix == "1h":
            continue
        for field in OHLC_FIELDS:
            col = f"{field}_{suffix}"
            base_col = f"{field}_1h"
            if col in merged and base_col in merged:
                merged[col] = merged[col].fillna(merged[base_col])

    if "close_1h" in merged.columns:
        merged["price_1h"] = merged["close_1h"]
    if "volume_1h" not in merged:
        merged["volume_1h"] = 0.0

    merged = _ensure_feature_columns(merged)
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged


def main():
    raw_dir = Path("data/raw")
    hist_file = Path("data/historical_data.csv")
    tickers_path = Path("data/tickers.txt")

    if tickers_path.exists():
        with open(tickers_path) as f:
            tickers = [t.strip() for t in f if t.strip()]
    else:
        tickers = ["TSLA"]
        logger.warning("tickers.txt not found. Using default ['TSLA'].")

    all_data = []
    for ticker in tickers:
        df_ticker = compute_for_ticker(raw_dir, ticker)
        if not df_ticker.empty:
            all_data.append(df_ticker)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        save_dataframe_with_timestamp_validation(
            final_df,
            hist_file,
            quoting=csv.QUOTE_MINIMAL,
            logger=logger,
        )
        logger.info(f"Saved historical data with indicators to {hist_file}")
    else:
        logger.error("No data computed. Check logs.")


if __name__ == '__main__':
    main()
