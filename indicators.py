# indicators.py
from __future__ import annotations
import numpy as np
import pandas as pd
import logging
import os
import re
from collections.abc import Callable, Mapping, Iterable

from fibonacci_utils import parse_fibonacci_levels


def _to_series(*args):
    out = []
    for x in args:
        if isinstance(x, pd.Series):
            out.append(x.astype(float))
        else:
            out.append(pd.Series(x, dtype=float))
    return out


def is_bullish_engulfing(open_, high, low, close):
    prev_open = open_.shift(1)
    prev_close = close.shift(1)
    open_, close, prev_open, prev_close = _to_series(open_, close, prev_open, prev_close)
    return ((prev_close < prev_open) & (close > open_) & (close > prev_open) & (open_ < prev_close)).fillna(False).astype(int)


def is_bearish_engulfing(open_, high, low, close):
    prev_open = open_.shift(1)
    prev_close = close.shift(1)
    open_, close, prev_open, prev_close = _to_series(open_, close, prev_open, prev_close)
    return ((prev_close > prev_open) & (close < open_) & (close < prev_open) & (open_ > prev_close)).fillna(False).astype(int)


def is_hammer(open_, high, low, close):
    open_, high, low, close = _to_series(open_, high, low, close)
    body = (close - open_).abs()
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    return ((lower_wick >= 2 * body) & (upper_wick <= body * 0.5)).fillna(False).astype(int)


def is_shooting_star(open_, high, low, close):
    open_, high, low, close = _to_series(open_, high, low, close)
    body = (close - open_).abs()
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    return ((upper_wick >= 2 * body) & (lower_wick <= body * 0.5)).fillna(False).astype(int)


def is_marubozu_bull(open_, high, low, close):
    open_, high, low, close = _to_series(open_, high, low, close)
    body = close - open_
    total_range = high - low + 1e-8
    return (
        (body > 0)
        & (body >= 0.95 * total_range)
        & (high - close <= body * 0.01)
        & (open_ - low <= body * 0.01)
    ).fillna(False).astype(int)


def is_marubozu_bear(open_, high, low, close):
    open_, high, low, close = _to_series(open_, high, low, close)
    body = open_ - close
    total_range = high - low + 1e-8
    return (
        (body > 0)
        & (body >= 0.95 * total_range)
        & (high - open_ <= body * 0.01)
        & (close - low <= body * 0.01)
    ).fillna(False).astype(int)


def is_morning_star(open_, high, low, close):
    o2, c2 = open_.shift(2), close.shift(2)
    o1, c1 = open_.shift(1), close.shift(1)
    o, c = open_, close
    return (
        (c2 < o2)
        & (abs(c1 - o1) < 0.3 * abs(c2 - o2))
        & (o1 < c2)
        & (c > o)
        & (c > (o2 + c2) / 2)
    ).fillna(False).astype(int)


def is_evening_star(open_, high, low, close):
    o2, c2 = open_.shift(2), close.shift(2)
    o1, c1 = open_.shift(1), close.shift(1)
    o, c = open_, close
    return (
        (c2 > o2)
        & (abs(c1 - o1) < 0.3 * abs(c2 - o2))
        & (o1 > c2)
        & (c < o)
        & (c < (o2 + c2) / 2)
    ).fillna(False).astype(int)

_EPSILON = 1e-8


def fast_candlestick_patterns(df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """Return a vectorised set of candlestick pattern signals with optional suffix."""

    patterns = get_candlestick_patterns(df, suffix=suffix)

    if patterns.empty:
        column_names = [f"{col}{suffix}" for col in DEFAULT_CANDLESTICK_PATTERN_COLUMNS]
        index = df.index if isinstance(df, pd.DataFrame) else pd.Index([])
        return pd.DataFrame(0, index=index, columns=column_names)

    expected_cols = [f"{col}{suffix}" for col in DEFAULT_CANDLESTICK_PATTERN_COLUMNS]
    return patterns.reindex(columns=expected_cols, fill_value=0).astype(int)


def _as_numpy(*arrays: np.ndarray) -> list[np.ndarray]:
    """Return the provided iterables as ``float`` numpy arrays."""

    return [np.asarray(array, dtype=float) for array in arrays]


def _resolve_close_series(data, column: str = "close") -> pd.Series | None:
    """Return a ``close`` series from ``data`` when possible."""

    if isinstance(data, pd.Series):
        return data
    if isinstance(data, pd.DataFrame):
        if column in data.columns:
            return data[column]
        return None
    return None


def _empty_pattern_result(length: int) -> np.ndarray:
    """Return an integer numpy array of zeros for ``length`` candles."""

    return np.zeros(length, dtype=int)


def _candle_components(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return real body, upper shadow, lower shadow and total range for candles."""

    real_body = np.abs(closes - opens)
    total_range = np.maximum(highs - lows, _EPSILON)
    upper_shadow = np.clip(highs - np.maximum(opens, closes), 0.0, None)
    lower_shadow = np.clip(np.minimum(opens, closes) - lows, 0.0, None)
    return real_body, upper_shadow, lower_shadow, total_range


def _fallback_cdlhammer(opens, highs, lows, closes):
    opens, highs, lows, closes = _as_numpy(opens, highs, lows, closes)
    length = len(opens)
    result = _empty_pattern_result(length)
    if length == 0:
        return result

    real_body, upper_shadow, lower_shadow, total_range = _candle_components(
        opens, highs, lows, closes
    )
    body_significant = real_body >= total_range * 0.05
    small_body = real_body <= total_range * 0.4
    long_lower = lower_shadow >= real_body * 2
    small_upper = upper_shadow <= real_body * 0.5
    bullish_close = closes >= opens

    mask = body_significant & small_body & long_lower & small_upper & bullish_close
    result[mask] = 100
    return result


def _fallback_cdlhangingman(opens, highs, lows, closes):
    opens, highs, lows, closes = _as_numpy(opens, highs, lows, closes)
    length = len(opens)
    result = _empty_pattern_result(length)
    if length == 0:
        return result

    real_body, upper_shadow, lower_shadow, total_range = _candle_components(
        opens, highs, lows, closes
    )
    body_significant = real_body >= total_range * 0.05
    small_body = real_body <= total_range * 0.4
    long_lower = lower_shadow >= real_body * 2
    small_upper = upper_shadow <= real_body * 0.5
    bearish_close = closes <= opens

    mask = body_significant & small_body & long_lower & small_upper & bearish_close
    result[mask] = -100
    return result


def _fallback_cdlinvertedhammer(opens, highs, lows, closes):
    opens, highs, lows, closes = _as_numpy(opens, highs, lows, closes)
    length = len(opens)
    result = _empty_pattern_result(length)
    if length == 0:
        return result

    real_body, upper_shadow, lower_shadow, total_range = _candle_components(
        opens, highs, lows, closes
    )
    body_significant = real_body >= total_range * 0.05
    small_body = real_body <= total_range * 0.4
    long_upper = upper_shadow >= real_body * 2
    small_lower = lower_shadow <= real_body * 0.5
    bullish_close = closes >= opens

    mask = body_significant & small_body & long_upper & small_lower & bullish_close
    result[mask] = 100
    return result


def _fallback_cdlshootingstar(opens, highs, lows, closes):
    opens, highs, lows, closes = _as_numpy(opens, highs, lows, closes)
    length = len(opens)
    result = _empty_pattern_result(length)
    if length == 0:
        return result

    real_body, upper_shadow, lower_shadow, total_range = _candle_components(
        opens, highs, lows, closes
    )
    body_significant = real_body >= total_range * 0.05
    small_body = real_body <= total_range * 0.4
    long_upper = upper_shadow >= real_body * 2
    small_lower = lower_shadow <= real_body * 0.5
    bearish_close = closes <= opens

    mask = body_significant & small_body & long_upper & small_lower & bearish_close
    result[mask] = -100
    return result


def _fallback_cdlengulfing(opens, highs, lows, closes):
    del highs, lows  # unused but kept for interface compatibility
    opens, closes = _as_numpy(opens, closes)
    length = len(opens)
    result = _empty_pattern_result(length)
    if length < 2:
        return result

    for idx in range(1, length):
        prev_open = opens[idx - 1]
        prev_close = closes[idx - 1]
        curr_open = opens[idx]
        curr_close = closes[idx]

        prev_bullish = prev_close > prev_open
        prev_bearish = prev_close < prev_open
        curr_bullish = curr_close > curr_open
        curr_bearish = curr_close < curr_open

        prev_body = max(abs(prev_close - prev_open), _EPSILON)
        curr_body = max(abs(curr_close - curr_open), _EPSILON)

        if curr_bullish and prev_bearish:
            if curr_open <= prev_close and curr_close >= prev_open and curr_body >= prev_body * 0.9:
                result[idx] = 100
        elif curr_bearish and prev_bullish:
            if curr_open >= prev_close and curr_close <= prev_open and curr_body >= prev_body * 0.9:
                result[idx] = -100

    return result


def _fallback_cdlpiercing(opens, highs, lows, closes):
    del highs, lows  # unused but kept for interface compatibility
    opens, closes = _as_numpy(opens, closes)
    length = len(opens)
    result = _empty_pattern_result(length)
    if length < 2:
        return result

    for idx in range(1, length):
        prev_open = opens[idx - 1]
        prev_close = closes[idx - 1]
        curr_open = opens[idx]
        curr_close = closes[idx]

        prev_bearish = prev_close < prev_open
        curr_bullish = curr_close > curr_open
        midpoint = (prev_open + prev_close) / 2

        if (
            prev_bearish
            and curr_bullish
            and curr_open < prev_close
            and curr_close > midpoint
            and curr_close < prev_open
        ):
            result[idx] = 100

    return result


def _fallback_cdlmorningstar(opens, highs, lows, closes):
    del highs, lows  # unused but kept for interface compatibility
    opens, closes = _as_numpy(opens, closes)
    length = len(opens)
    result = _empty_pattern_result(length)
    if length < 3:
        return result

    for idx in range(2, length):
        first_open = opens[idx - 2]
        first_close = closes[idx - 2]
        second_open = opens[idx - 1]
        second_close = closes[idx - 1]
        third_open = opens[idx]
        third_close = closes[idx]

        first_bearish = first_close < first_open
        third_bullish = third_close > third_open

        first_body = abs(first_close - first_open)
        second_body = abs(second_close - second_open)

        midpoint = (first_open + first_close) / 2

        if not (first_bearish and third_bullish):
            continue

        gap_down = second_open < first_close and second_close < first_close
        small_second = second_body <= max(first_body * 0.6, first_body * 0.1)
        strong_finish = third_close >= midpoint

        if gap_down and small_second and strong_finish:
            result[idx] = 100

    return result


def _fallback_cdleveningstar(opens, highs, lows, closes):
    del highs, lows  # unused but kept for interface compatibility
    opens, closes = _as_numpy(opens, closes)
    length = len(opens)
    result = _empty_pattern_result(length)
    if length < 3:
        return result

    for idx in range(2, length):
        first_open = opens[idx - 2]
        first_close = closes[idx - 2]
        second_open = opens[idx - 1]
        second_close = closes[idx - 1]
        third_open = opens[idx]
        third_close = closes[idx]

        first_bullish = first_close > first_open
        third_bearish = third_close < third_open

        first_body = abs(first_close - first_open)
        second_body = abs(second_close - second_open)

        midpoint = (first_open + first_close) / 2

        if not (first_bullish and third_bearish):
            continue

        gap_up = second_open > first_close and second_close > first_close
        small_second = second_body <= max(first_body * 0.6, first_body * 0.1)
        strong_finish = third_close <= midpoint

        if gap_up and small_second and strong_finish:
            result[idx] = -100

    return result


def _fallback_cdl3whitesoldiers(opens, highs, lows, closes):
    opens, highs, lows, closes = _as_numpy(opens, highs, lows, closes)
    length = len(opens)
    result = _empty_pattern_result(length)
    if length < 3:
        return result

    real_body, _, _, total_range = _candle_components(opens, highs, lows, closes)

    for idx in range(2, length):
        o1, o2, o3 = opens[idx - 2], opens[idx - 1], opens[idx]
        c1, c2, c3 = closes[idx - 2], closes[idx - 1], closes[idx]
        r1, r2, r3 = real_body[idx - 2], real_body[idx - 1], real_body[idx]
        tr1, tr2, tr3 = total_range[idx - 2], total_range[idx - 1], total_range[idx]

        bull1 = c1 > o1
        bull2 = c2 > o2
        bull3 = c3 > o3

        if not (bull1 and bull2 and bull3):
            continue

        long_bodies = all(
            body >= rng * 0.5 if rng > _EPSILON else body > _EPSILON
            for body, rng in ((r1, tr1), (r2, tr2), (r3, tr3))
        )
        opens_within = (o2 > min(o1, c1)) and (o2 < max(o1, c1)) and (o3 > min(o2, c2)) and (o3 < max(o2, c2))
        closes_higher = (c2 > c1) and (c3 > c2)

        if long_bodies and opens_within and closes_higher:
            result[idx] = 100

    return result


def _fallback_cdl3blackcrows(opens, highs, lows, closes):
    opens, highs, lows, closes = _as_numpy(opens, highs, lows, closes)
    length = len(opens)
    result = _empty_pattern_result(length)
    if length < 3:
        return result

    real_body, _, _, total_range = _candle_components(opens, highs, lows, closes)

    for idx in range(2, length):
        o1, o2, o3 = opens[idx - 2], opens[idx - 1], opens[idx]
        c1, c2, c3 = closes[idx - 2], closes[idx - 1], closes[idx]
        r1, r2, r3 = real_body[idx - 2], real_body[idx - 1], real_body[idx]
        tr1, tr2, tr3 = total_range[idx - 2], total_range[idx - 1], total_range[idx]

        bear1 = c1 < o1
        bear2 = c2 < o2
        bear3 = c3 < o3

        if not (bear1 and bear2 and bear3):
            continue

        long_bodies = all(
            body >= rng * 0.5 if rng > _EPSILON else body > _EPSILON
            for body, rng in ((r1, tr1), (r2, tr2), (r3, tr3))
        )
        opens_within = (o2 < max(o1, c1)) and (o2 > min(o1, c1)) and (o3 < max(o2, c2)) and (o3 > min(o2, c2))
        closes_lower = (c2 < c1) and (c3 < c2)

        if long_bodies and opens_within and closes_lower:
            result[idx] = -100

    return result


_FALLBACK_PATTERN_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "CDLHAMMER": _fallback_cdlhammer,
    "CDLHANGINGMAN": _fallback_cdlhangingman,
    "CDLINVERTEDHAMMER": _fallback_cdlinvertedhammer,
    "CDLSHOOTINGSTAR": _fallback_cdlshootingstar,
    "CDLENGULFING": _fallback_cdlengulfing,
    "CDLPIERCING": _fallback_cdlpiercing,
    "CDLMORNINGSTAR": _fallback_cdlmorningstar,
    "CDLEVENINGSTAR": _fallback_cdleveningstar,
    "CDL3WHITESOLDIERS": _fallback_cdl3whitesoldiers,
    "CDL3BLACKCROWS": _fallback_cdl3blackcrows,
}


class _TaLibStub:
    """Fallback stub that emulates key candlestick pattern functions."""

    def __getattr__(self, name: str):
        if name in _FALLBACK_PATTERN_FUNCTIONS:
            return _FALLBACK_PATTERN_FUNCTIONS[name]

        def _stub(*args, **kwargs):
            length = len(args[0]) if args else 0
            return np.zeros(length, dtype=int)

        return _stub


talib = _TaLibStub()
_TA_LIB_AVAILABLE = False

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/indicators.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info(
    "Using built-in heuristic candlestick pattern approximations for candlestick analysis."
)


def _slugify_pattern_name(name: str) -> str:
    """Return a snake_case feature suffix derived from ``name``."""

    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# === CANDLESTICK PATTERN DEFAULT FEATURES ===
# These are common patterns used in ML models
DEFAULT_CANDLESTICK_PATTERN_FEATURES: list[str] = [
    "CDLDOJI",
    "CDLENGULFING",
    "CDLHARAMI",
    "CDLHARAMICROSS",
    "CDLMORNINGSTAR",
    "CDLEVENINGSTAR",
    "CDL3WHITESOLDIERS",
    "CDL3BLACKCROWS",
    "CDLHANGINGMAN",
    "CDLSPINNINGTOP",
    "CDLSHOOTINGSTAR",
    "CDLHAMMER",
    "CDLINVERTEDHAMMER",
    "CDLMARUBOZU",
    "CDLDRAGONFLYDOJI",
    "CDLGRAVESTONEDOJI",
    "CDLLONGLEGGEDDOJI",
    "CDLRICKSHAWMAN",
    "CDLBELTHOLD",
    "CDLHIKKAKE",
    "CDLPIERCING",
    "CDLDARKCLOUDCOVER",
    "CDL3INSIDE",
    "CDL3OUTSIDE",
    "CDLKICKING",
    "CDLABANDONEDBABY",
    "CDLADVANCEBLOCK",
    "CDLBREAKAWAY",
    "CDLCLOSINGMARUBOZU",
    "CDLCONCEALBABYSWALL",
    "CDLCOUNTERATTACK",
    "CDLGAPSIDESIDEWHITE",
    "CDLINNECK",
    "CDLLADDERBOTTOM",
    "CDLMATHOLD",
    "CDLRISEFALL3METHODS",
    "CDLSEPARATINGLINES",
    "CDLSTALLEDPATTERN",
    "CDLSTICKSANDWICH",
    "CDLTAKURI",
    "CDLTASUKIGAP",
    "CDLTHRUSTING",
    "CDLTRISTAR",
    "CDLUNIQUE3RIVER",
    "CDLUPSIDEGAP2CROWS",
    "CDLXSIDEGAP3METHODS",
]

_CANDLESTICK_PATTERN_NAME_MAP: dict[str, str] = {
    "CDLDOJI": "Doji",
    "CDLENGULFING": "Engulfing",
    "CDLHARAMI": "Harami",
    "CDLHARAMICROSS": "Harami Cross",
    "CDLMORNINGSTAR": "Morning Star",
    "CDLEVENINGSTAR": "Evening Star",
    "CDL3WHITESOLDIERS": "Three White Soldiers",
    "CDL3BLACKCROWS": "Three Black Crows",
    "CDLHANGINGMAN": "Hanging Man",
    "CDLSPINNINGTOP": "Spinning Top",
    "CDLSHOOTINGSTAR": "Shooting Star",
    "CDLHAMMER": "Hammer",
    "CDLINVERTEDHAMMER": "Inverted Hammer",
    "CDLMARUBOZU": "Marubozu",
    "CDLDRAGONFLYDOJI": "Dragonfly Doji",
    "CDLGRAVESTONEDOJI": "Gravestone Doji",
    "CDLLONGLEGGEDDOJI": "Long-Legged Doji",
    "CDLRICKSHAWMAN": "Rickshaw Man",
    "CDLBELTHOLD": "Belt Hold",
    "CDLHIKKAKE": "Hikkake",
    "CDLPIERCING": "Piercing Line",
    "CDLDARKCLOUDCOVER": "Dark Cloud Cover",
    "CDL3INSIDE": "Three Inside",
    "CDL3OUTSIDE": "Three Outside",
    "CDLKICKING": "Kicking",
    "CDLABANDONEDBABY": "Abandoned Baby",
    "CDLADVANCEBLOCK": "Advance Block",
    "CDLBREAKAWAY": "Breakaway",
    "CDLCLOSINGMARUBOZU": "Closing Marubozu",
    "CDLCONCEALBABYSWALL": "Concealing Baby Swallow",
    "CDLCOUNTERATTACK": "Counterattack",
    "CDLGAPSIDESIDEWHITE": "Gap Side-by-Side White",
    "CDLINNECK": "In-Neck",
    "CDLLADDERBOTTOM": "Ladder Bottom",
    "CDLMATHOLD": "Mat Hold",
    "CDLRISEFALL3METHODS": "Rise/Fall Three Methods",
    "CDLSEPARATINGLINES": "Separating Lines",
    "CDLSTALLEDPATTERN": "Stalled Pattern",
    "CDLSTICKSANDWICH": "Stick Sandwich",
    "CDLTAKURI": "Takuri",
    "CDLTASUKIGAP": "Tasuki Gap",
    "CDLTHRUSTING": "Thrusting",
    "CDLTRISTAR": "Tristar",
    "CDLUNIQUE3RIVER": "Unique Three River",
    "CDLUPSIDEGAP2CROWS": "Upside Gap Two Crows",
    "CDLXSIDEGAP3METHODS": "Upside/Downside Gap Three Methods",
}


def _pattern_display_name(pattern: str) -> str:
    return _CANDLESTICK_PATTERN_NAME_MAP.get(pattern, pattern)


CANDLESTICK_PATTERN_DEFINITIONS: list[tuple[str, str]] = [
    ("CDLHAMMER", _pattern_display_name("CDLHAMMER")),
    ("CDLINVERTEDHAMMER", _pattern_display_name("CDLINVERTEDHAMMER")),
    ("CDLENGULFING", _pattern_display_name("CDLENGULFING")),
    ("CDLPIERCING", _pattern_display_name("CDLPIERCING")),
    ("CDLMORNINGSTAR", _pattern_display_name("CDLMORNINGSTAR")),
    ("CDL3WHITESOLDIERS", _pattern_display_name("CDL3WHITESOLDIERS")),
    ("CDLHANGINGMAN", _pattern_display_name("CDLHANGINGMAN")),
    ("CDLSHOOTINGSTAR", _pattern_display_name("CDLSHOOTINGSTAR")),
    ("CDLEVENINGSTAR", _pattern_display_name("CDLEVENINGSTAR")),
    ("CDL3BLACKCROWS", _pattern_display_name("CDL3BLACKCROWS")),
    ("CDLDARKCLOUDCOVER", _pattern_display_name("CDLDARKCLOUDCOVER")),
    ("CDLDOJI", _pattern_display_name("CDLDOJI")),
    ("CDLSPINNINGTOP", _pattern_display_name("CDLSPINNINGTOP")),
    ("CDLFALLINGTHREEMETHODS", _pattern_display_name("CDLFALLINGTHREEMETHODS")),
    ("CDLRISINGTHREEMETHODS", _pattern_display_name("CDLRISINGTHREEMETHODS")),
]

# Focus on the high-signal patterns used throughout the training pipeline.
DEFAULT_CANDLESTICK_PATTERN_SPECS: list[tuple[str, str, str]] = [
    ("Hammer", "pattern_hammer", "CDLHAMMER"),
    ("Inverted Hammer", "pattern_inverted_hammer", "CDLINVERTEDHAMMER"),
    ("Engulfing", "pattern_engulfing", "CDLENGULFING"),
    ("Piercing Line", "pattern_piercing_line", "CDLPIERCING"),
    ("Morning Star", "pattern_morning_star", "CDLMORNINGSTAR"),
    ("Three White Soldiers", "pattern_three_white_soldiers", "CDL3WHITESOLDIERS"),
    ("Hanging Man", "pattern_hanging_man", "CDLHANGINGMAN"),
    ("Shooting Star", "pattern_shooting_star", "CDLSHOOTINGSTAR"),
    ("Evening Star", "pattern_evening_star", "CDLEVENINGSTAR"),
    ("Three Black Crows", "pattern_three_black_crows", "CDL3BLACKCROWS"),
]

DEFAULT_CANDLESTICK_PATTERN_NAMES: list[str] = [
    name for name, _, _ in DEFAULT_CANDLESTICK_PATTERN_SPECS
]
DEFAULT_CANDLESTICK_PATTERN_COLUMNS: list[str] = [
    column for _, column, _ in DEFAULT_CANDLESTICK_PATTERN_SPECS
]
DEFAULT_CANDLESTICK_PATTERN_CODES: list[str] = [
    code for _, _, code in DEFAULT_CANDLESTICK_PATTERN_SPECS
]

def calculate_rsi(df, period=14):
    """Calculate the Relative Strength Index (RSI) for the given dataframe.

    If zero loss is detected in the calculation window, the function reuses the
    last valid RSI value instead of returning a neutral 50. If no previous RSI
    value is available, ``None`` is returned.
    """
    if not isinstance(period, int) or period <= 0:
        logger.error(f"Invalid period: {period}. Must be a positive integer.")
        return None
    close_series = _resolve_close_series(df)
    if close_series is None or close_series.empty:
        logger.warning("Empty or invalid data for RSI calculation. Returning None.")
        return None
    try:
        delta = close_series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean() + 1e-6  # Add epsilon
        rs = pd.Series(np.where(loss <= 1e-6, np.nan, gain / loss), index=loss.index)
        rsi = 100 - (100 / (1 + rs))
        rsi_value = rsi.iloc[-1] if not rsi.empty else None
        if pd.isna(rsi_value):
            logger.warning("Zero loss detected in RSI calculation. Using last valid RSI value.")
            rsi_value = rsi.iloc[:-1].dropna().iloc[-1] if not rsi.iloc[:-1].dropna().empty else None
        logger.debug(f"RSI calculated: {rsi_value}")
        return rsi_value
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return None

def calculate_macd(df, fast=12, slow=26, signal_period=9):
    """Calculate MACD and Signal line for the given dataframe."""
    if any(not isinstance(p, int) or p <= 0 for p in [fast, slow, signal_period]):
        logger.error(f"Invalid MACD parameters: fast={fast}, slow={slow}, signal_period={signal_period}.")
        return None, None
    close_series = _resolve_close_series(df)
    if close_series is None or close_series.empty:
        logger.warning("Empty or invalid data for MACD calculation. Returning None.")
        return None, None
    try:
        exp1 = close_series.ewm(span=fast, adjust=False).mean()
        exp2 = close_series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        logger.debug(f"MACD: {macd.iloc[-1] if not macd.empty else None}, Signal: {signal.iloc[-1] if not signal.empty else None}")
        return macd.iloc[-1] if not macd.empty else None, signal.iloc[-1] if not signal.empty else None
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return None, None

def calculate_td_sequential(df):
    """Calculate TD Sequential setup counts.

    Returns
    -------
    tuple[list[int], list[int], int, int]
        A tuple containing the historical buy and sell setup counts along with
        the latest buy and sell counts for the most recent candle.
    """

    close_series = _resolve_close_series(df)
    if close_series is None or close_series.empty:
        logger.warning("Empty or invalid data for TD Sequential calculation. Returning defaults.")
        return [0], [0], 0, 0
    try:
        close = close_series.reset_index(drop=True)
        buy_setup = [0] * len(close)
        sell_setup = [0] * len(close)

        for i in range(4, len(close)):
            if close.iloc[i] < close.iloc[i - 4]:
                buy_setup[i] = buy_setup[i - 1] + 1
            else:
                buy_setup[i] = 0

            if close.iloc[i] > close.iloc[i - 4]:
                sell_setup[i] = sell_setup[i - 1] + 1
            else:
                sell_setup[i] = 0

        latest_buy = buy_setup[-1] if buy_setup else 0
        latest_sell = sell_setup[-1] if sell_setup else 0

        logger.debug(
            f"TD Sequential counts: Buy Setup={latest_buy}, Sell Setup={latest_sell}"
        )
        return buy_setup, sell_setup, latest_buy, latest_sell
    except Exception as e:
        logger.error(f"Error calculating TD Sequential: {e}")
        return [0], [0], 0, 0


def summarize_td_sequential(buy_count: int, sell_count: int) -> str:
    """Summarise TD Sequential counts for display purposes."""

    if buy_count > 0:
        return f"Buy {buy_count}"
    if sell_count > 0:
        return f"Sell {sell_count}"
    return "N/A"

def calculate_tds_trend(df, fact=2, num=6, length=3, limit=7, conti=False, filter_pct=100):
    """Approximate the Trend Direction Sequence (TDS) indicator.

    The original Pine Script calculates trend sequences across multiple
    higher timeframes.  This Python adaptation mirrors that behaviour by
    dynamically spacing comparisons based on ``length`` and ``fact``.  The
    most recent run of consecutive directional moves is measured for each
    timeframe and summarised into an overall trend percentage and signal.

    Parameters are intentionally similar to the TradingView version so the
    agent can experiment with different configurations.
    """

    # Basic validation of inputs
    if any(not isinstance(p, (int, float)) or p <= 0 for p in [fact, num, length, limit, filter_pct]):
        logger.error(
            "Invalid TDS Trend parameters: "
            f"fact={fact}, num={num}, length={length}, limit={limit}, filter_pct={filter_pct}"
        )
        return 0, 0
    close_series = _resolve_close_series(df)
    if close_series is None or close_series.empty:
        logger.warning("Empty or invalid data for TDS Trend calculation. Returning defaults.")
        return 0, 0

    try:
        closes = close_series
        trends: list[int] = []

        # Examine progressively larger "virtual" timeframes.
        for i in range(num + 1):
            step = int((fact ** i) * length)
            if step <= 0 or len(closes) <= step:
                trends.append(0)
                continue

            # Compute differences between closes separated by ``step`` bars.
            diff = closes.diff(step).dropna()
            if diff.empty:
                trends.append(0)
                continue

            direction = np.sign(diff.values)

            # Determine the most recent uninterrupted sequence length.
            run = 0
            for d in reversed(direction):
                if d > 0:
                    run = run + 1 if run >= 0 else 1
                elif d < 0:
                    run = run - 1 if run <= 0 else -1
                else:  # diff == 0 breaks the sequence
                    break
                if not conti and abs(run) >= limit:
                    break

            if abs(run) >= limit or (conti and run != 0):
                trends.append(int(np.sign(run)))
            else:
                trends.append(0)

        sum_trend = sum(trends)
        overall_trend = float(round(sum_trend / (num + 1) * 100, 0))

        signal = 0.0
        if overall_trend > filter_pct:
            signal = 1.0
        elif overall_trend < -filter_pct:
            signal = -1.0

        logger.debug(f"TDS Trend: {overall_trend}%, Signal: {signal}")
        return overall_trend, signal

    except Exception as e:
        logger.error(f"Error calculating TDS Trend: {e}")
        return 0, 0

def calculate_atr(df, period=10):
    """Calculate Average True Range (ATR) for the given dataframe."""
    if not isinstance(df, pd.DataFrame):
        logger.warning("ATR calculation expects a DataFrame with OHLC data. Returning None.")
        return None
    if df.empty or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
        logger.warning("Empty or invalid dataframe for ATR calculation. Returning None.")
        return None
    # Guard against insufficient or NaN data
    price_cols = df[['high', 'low', 'close']]
    if price_cols.isnull().any().any():
        logger.warning("NaN values found in price data for ATR calculation. Returning None.")
        return None
    if len(price_cols) < period:
        logger.warning(f"Not enough data points ({len(price_cols)}) for ATR period {period}. Returning None.")
        return None
    try:
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        last_atr = atr.iloc[-1] if not atr.empty else None
        logger.debug(f"ATR calculated: {last_atr}")
        return last_atr if pd.notna(last_atr) else None
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return None

def detect_pivots(df, window=5, high_col='high', low_col='low', **_):
    """Detect swing high/low pivots in ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        Price dataframe containing high/low columns.
    window : int, optional
        Rolling window size for local extrema detection.
    high_col : str, optional
        Column name for highs (defaults to ``'high'``).
    low_col : str, optional
        Column name for lows (defaults to ``'low'``).
    """
    if df.empty or high_col not in df.columns or low_col not in df.columns:
        return [
            {
                'time': df.index[-1] if not df.empty else pd.Timestamp.now(tz='UTC'),
                'price': df[high_col].mean() if not df.empty else 0,
                'is_high': True,
            }
        ] * 2  # Default 2 pivots

    # Rolling high/low for more pivots
    rolling_high = df[high_col].rolling(window=window).max()
    rolling_low = df[low_col].rolling(window=window).min()
    highs = (df[high_col] == rolling_high) & (df[high_col] > df[high_col].shift(1))
    lows = (df[low_col] == rolling_low) & (df[low_col] < df[low_col].shift(1))
    pivots = []
    for idx in highs[highs].index:
        pivots.append({'time': idx, 'price': df.loc[idx, 'high'], 'is_high': True})
    for idx in lows[lows].index:
        pivots.append({'time': idx, 'price': df.loc[idx, 'low'], 'is_high': False})

    if len(pivots) < 2:
        # Fallback to recent high/low
        recent_high = df[high_col].max()
        recent_low = df[low_col].min()
        pivots = [
            {'time': df.index[-1], 'price': recent_high, 'is_high': True},
            {'time': df.index[0], 'price': recent_low, 'is_high': False},
        ]
    return pivots

def calculate_fib_levels_from_pivots(pivots, levels=None):
    """Calculate Fibonacci levels (including negative levels) from pivots."""
    if levels is None:
        levels = [-0.618, -0.382, -0.236, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.414, 1.618, 1.65, 2.618, 2.65, 3.618, 3.65, 4.236, 4.618]
    
    if len(pivots) < 2:
        logger.warning(f"Insufficient pivots for Fibonacci calculation. Got {len(pivots)} pivots: {pivots}")
        return {}
    
    try:
        pivots = sorted(pivots, key=lambda p: p[0] if isinstance(p, tuple) else p['time'])
        logger.debug(f"Sorted pivots: {pivots}")
        last_pivot = pivots[-1]
        prev_pivot = pivots[-2]
        last_price = last_pivot[1] if isinstance(last_pivot, tuple) else last_pivot['price']
        prev_price = prev_pivot[1] if isinstance(prev_pivot, tuple) else prev_pivot['price']
        last_is_high = last_pivot[2] if isinstance(last_pivot, tuple) else last_pivot['is_high']
        prev_is_high = prev_pivot[2] if isinstance(prev_pivot, tuple) else prev_pivot['is_high']
        price_diff = last_price - prev_price
        is_retracement = last_is_high != prev_is_high

        fib_levels = {}
        for level in levels:
            if is_retracement:
                if last_is_high:
                    fib_price = last_price - abs(price_diff) * level
                else:
                    fib_price = last_price + abs(price_diff) * level
            else:
                if last_is_high:
                    fib_price = last_price + abs(price_diff) * level
                else:
                    fib_price = last_price - abs(price_diff) * level
            fib_levels[level] = fib_price

        logger.debug(f"Calculated Fibonacci levels: {fib_levels}")
        return fib_levels
    except Exception as e:
        logger.error(f"Error calculating Fibonacci levels: {e}")
        return {}

def parse_fibonacci(fib_str, price):
    """Parse Fibonacci levels from a summary string."""

    if pd.isna(fib_str) or fib_str in (None, "", "N/A"):
        logger.warning("Invalid Fibonacci string. Returning default [0] * 5.")
        return [0.0] * 5

    levels = parse_fibonacci_levels(fib_str, max_levels=5)
    if not any(levels):
        logger.warning("Skipping invalid Fibonacci summary. Returning default [0] * 5.")
        return [0.0] * 5
    return levels

def calculate_fib_time_zones(pivots, current_time, num_zones=10):
    """Calculate Fibonacci Time Zones from pivot points."""
    if not isinstance(num_zones, int) or num_zones <= 0:
        logger.error(f"Invalid num_zones: {num_zones}. Must be a positive integer.")
        return []
    if len(pivots) < 2:
        logger.warning(f"Insufficient pivots for Fibonacci time zones. Got {len(pivots)} pivots: {pivots}")
        return []
    try:
        time_last = pivots[-1][0] if isinstance(pivots[-1], tuple) else pivots[-1]['time']
        time_prev = pivots[-2][0] if isinstance(pivots[-2], tuple) else pivots[-2]['time']
        time_diff = (time_last - time_prev).total_seconds() if isinstance(time_last, pd.Timestamp) else (time_last - time_prev)
        fib_ratios = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        fib_zones = fib_ratios[:num_zones]
        zones = [time_last + pd.Timedelta(seconds=time_diff * ratio) if isinstance(time_last, pd.Timestamp) else time_last + time_diff * ratio for ratio in fib_zones]
        logger.debug(f"Fibonacci time zones: {zones}")
        return zones
    except Exception as e:
        logger.error(f"Error calculating Fibonacci time zones: {e}")
        return []

def calculate_pivot_points(df, levels=None):
    """Calculate Fibonacci-style pivot point support and resistance levels."""
    if levels is None:
        levels = [0.382, 0.618, 1.0]
    if not isinstance(df, pd.DataFrame):
        logger.warning("Pivot point calculation expects a DataFrame with OHLC data. Returning empty dict.")
        return {}
    if df.empty or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
        logger.warning("Empty or invalid dataframe for pivot point calculation. Returning empty dict.")
        return {}
    try:
        row = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
        high = float(row['high'])
        low = float(row['low'])
        close = float(row['close'])
        pivot = (high + low + close) / 3.0
        rng = high - low
        pivots = {'pivot': pivot}
        for idx, level in enumerate(levels, 1):
            pivots[f'R{idx}'] = pivot + rng * level
            pivots[f'S{idx}'] = pivot - rng * level
        logger.debug(f"Pivot points: {pivots}")
        return pivots
    except Exception as e:
        logger.error(f"Error calculating pivot points: {e}")
        return {}

def _extract_pivot_prices(value) -> list[float]:
    if value in (None, "", "N/A"):
        return []
    if isinstance(value, Mapping):
        values = value.values()
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        values = value
    elif isinstance(value, str):
        matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        return [float(match) for match in matches if match]
    else:
        values = [value]

    prices: list[float] = []
    for item in values:
        try:
            numeric = float(item)
        except (TypeError, ValueError):
            continue
        if np.isnan(numeric):
            continue
        prices.append(numeric)
    return prices


def calculate_level_weight(
    source,
    fib_levels=None,
    pivot_points=None,
    tolerance=0.005,
    base_weight=1.0,
    hit_weight=1.5,
    timeframe: str | None = None,
):
    """Return a higher weight when ``price`` is near key Fibonacci or pivot levels.

    Parameters
    ----------
    source : float | Mapping[str, object]
        Either the current price or an indicator mapping containing Fibonacci
        summaries and pivot points. When ``source`` is a mapping the next
        positional argument is interpreted as the price.
    fib_levels : list[float] | None
        Iterable of Fibonacci price levels.
    pivot_points : list[float] | None
        Iterable of pivot point price levels.
    tolerance : float, optional
        Relative tolerance (as a fraction of price) to consider a "hit".
    base_weight : float, optional
        Weight returned when no level is hit.
    hit_weight : float, optional
        Weight returned when price is within ``tolerance`` of any level.
    """
    if isinstance(source, pd.DataFrame):
        df = source

        def _candidate_columns(base: str) -> list[str]:
            cols = []
            if timeframe:
                cols.append(f"{base}_{timeframe}")
            cols.append(base)
            return cols

        price_candidates = []
        price_candidates.extend(_candidate_columns("price"))
        price_candidates.extend(_candidate_columns("close"))
        fib_candidates = _candidate_columns("fib_summary")
        pivot_candidates = _candidate_columns("pivot_points")

        def _coerce_float(value) -> float:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return 0.0
            return 0.0 if not np.isfinite(numeric) else numeric

        def _select_value(row: pd.Series, candidates: list[str]):
            for column in candidates:
                if column in row.index:
                    value = row[column]
                    if value in (None, "", "N/A"):
                        continue
                    if isinstance(value, (int, float)) and np.isnan(value):
                        continue
                    return value
            return None

        def _row_weight(row: pd.Series) -> float:
            price_value = _coerce_float(_select_value(row, price_candidates))
            fib_summary = _select_value(row, fib_candidates)
            fib_prices = parse_fibonacci(fib_summary, price_value) if fib_summary not in (None, "", "N/A") else []
            pivot_value = _select_value(row, pivot_candidates)
            pivot_prices = _extract_pivot_prices(pivot_value)
            return float(
                calculate_level_weight(
                    price_value,
                    fib_prices,
                    pivot_prices,
                    tolerance=tolerance,
                    base_weight=base_weight,
                    hit_weight=hit_weight,
                )
            )

        if df.empty:
            return pd.Series(dtype=float)

        return df.apply(_row_weight, axis=1)

    indicators: Mapping[str, object] | None = None
    if isinstance(source, Mapping):
        indicators = source
        try:
            price = float(fib_levels) if fib_levels is not None else 0.0
        except (TypeError, ValueError):
            price = 0.0
        fib_values: list[float] = []
        fib_map = indicators.get("fib_summary", {}) if isinstance(indicators, Mapping) else {}
        if isinstance(fib_map, Mapping):
            for summary in fib_map.values():
                try:
                    fib_values.extend(parse_fibonacci(summary, price))
                except Exception:
                    continue
        pivot_values: list[float] = []
        pivot_map = indicators.get("pivot_points", {}) if isinstance(indicators, Mapping) else {}
        if isinstance(pivot_map, Mapping):
            for pivot_dict in pivot_map.values():
                if isinstance(pivot_dict, Mapping):
                    pivot_values.extend(_extract_pivot_prices(pivot_dict))
        fib_levels = fib_values
        pivot_points = pivot_values
    else:
        try:
            price = float(source)
        except (TypeError, ValueError):
            price = 0.0

    def _ensure_iterable(values):
        """Return a flat list of numeric levels from ``values``.

        ``values`` may be ``None``, a scalar, mapping, or iterable. Any values
        that cannot be converted to finite floats are ignored.
        """

        if values is None:
            return []

        if isinstance(values, Mapping):
            values = values.values()

        if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
            items = values
        else:
            items = [values]

        cleaned: list[float] = []
        for item in items:
            try:
                numeric = float(item)
            except (TypeError, ValueError):
                continue
            if np.isnan(numeric):
                continue
            cleaned.append(numeric)
        return cleaned

    fib_levels = _ensure_iterable(fib_levels)
    pivot_points = _ensure_iterable(pivot_points)
    levels = [lvl for lvl in fib_levels if lvl] + [lvl for lvl in pivot_points if lvl]
    for lvl in levels:
        try:
            if lvl and abs(price - lvl) / lvl <= tolerance:
                logger.debug(f"Price {price} hit level {lvl} within tolerance {tolerance}")
                return hit_weight
        except Exception as e:  # pragma: no cover - defensive programming
            logger.warning(f"Failed level weight comparison for {lvl}: {e}")
    return base_weight

def calculate_zig_zag(df, deviation_multiplier=2.0, depth=5):
    """Calculate Zig Zag lines connecting pivot highs and lows."""
    if any(not isinstance(p, (int, float)) or p <= 0 for p in [depth]):
        logger.error(f"Invalid Zig Zag parameters: deviation_multiplier={deviation_multiplier}, depth={depth}")
        return []
    if not isinstance(df, pd.DataFrame):
        logger.warning("Zig Zag calculation expects a DataFrame with OHLC data. Returning empty list.")
        return []
    if df.empty or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
        logger.warning("Empty or invalid dataframe for Zig Zag calculation. Returning empty list.")
        return []
    try:
        atr = calculate_atr(df, period=10)
        deviation = atr / df['close'].iloc[-1] * 100 * deviation_multiplier if atr is not None and df['close'].iloc[-1] != 0 else 0.5
        logger.debug(f"Zig Zag ATR-based deviation: {deviation:.2f}%")
        
        pivots = detect_pivots(df, window=depth)
        pivots_high = [p for p in pivots if p['is_high']]
        pivots_low = [p for p in pivots if not p['is_high']]
        zig_zag_lines = []
        last_pivot = None
        for pivot in sorted(pivots_high + pivots_low, key=lambda p: p[0] if isinstance(p, tuple) else p['time']):
            if last_pivot:
                time1 = last_pivot[0] if isinstance(last_pivot, tuple) else last_pivot['time']
                price1 = last_pivot[1] if isinstance(last_pivot, tuple) else last_pivot['price']
                time2 = pivot[0] if isinstance(pivot, tuple) else pivot['time']
                price2 = pivot[1] if isinstance(pivot, tuple) else pivot['price']
                zig_zag_lines.append((time1, price1, time2, price2))
            last_pivot = pivot
        logger.debug(f"Zig Zag lines: {zig_zag_lines}")
        return zig_zag_lines
    except Exception as e:
        logger.error(f"Error calculating Zig Zag: {e}")
        return []

def calculate_high_volatility(df, atr_period=13, multiplier=2.718):
    """Detect high volatility bars based on pandas-calculated ATR."""
    if any(not isinstance(p, (int, float)) or p <= 0 for p in [atr_period, multiplier]):
        logger.error(f"Invalid volatility parameters: atr_period={atr_period}, multiplier={multiplier}")
        return False
    if not isinstance(df, pd.DataFrame):
        logger.warning("Volatility calculation expects a DataFrame with OHLC data. Returning False.")
        return False
    if df.empty or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
        logger.warning("Empty or invalid dataframe for volatility calculation. Returning False.")
        return False
    try:
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_period).mean()
        range_bar = df['high'] - df['low']
        high_vol = range_bar > atr * multiplier
        logger.debug(f"High volatility: {bool(high_vol.iloc[-1]) if not high_vol.empty else False}")
        return bool(high_vol.iloc[-1]) if not high_vol.empty else False
    except Exception as e:
        logger.error(f"Error calculating high volatility: {e}")
        return False

def calculate_volume_spike(df, sma_period=89, threshold=4.669):
    """Detect volume spikes indicating potential exhaustion.

    Accepts either a DataFrame with a ``volume`` column or a ``Series`` of
    volume values, allowing downstream callers to pass whichever is available.
    """
    if any(not isinstance(p, (int, float)) or p <= 0 for p in [sma_period, threshold]):
        logger.error(f"Invalid volume spike parameters: sma_period={sma_period}, threshold={threshold}")
        return False

    # Normalize inputs to a Series for calculation
    if isinstance(df, pd.Series):
        volume_series = df
    elif isinstance(df, pd.DataFrame) and 'volume' in df.columns:
        volume_series = df['volume']
    else:
        logger.warning("Empty or invalid dataframe for volume spike calculation. Returning False.")
        return False

    if volume_series.empty:
        logger.warning("Empty volume series for volume spike calculation. Returning False.")
        return False

    try:
        vol_sma = volume_series.rolling(window=sma_period).mean()
        spike = volume_series > vol_sma * threshold
        logger.debug(f"Volume spike: {bool(spike.iloc[-1]) if not spike.empty else False}")
        return bool(spike.iloc[-1]) if not spike.empty else False
    except Exception as e:
        logger.error(f"Error calculating volume spike: {e}")
        return False

def calculate_volume_weighted_category(df, sma_period=89, high_thresh=1.618, low_thresh=0.618):
    """Categorize volume strength based on SMA and candle direction."""
    if any(not isinstance(p, (int, float)) or p <= 0 for p in [sma_period, high_thresh, low_thresh]):
        logger.error(f"Invalid volume category parameters: sma_period={sma_period}, high_thresh={high_thresh}, low_thresh={low_thresh}")
        return "Neutral Volume"
    if not isinstance(df, pd.DataFrame):
        logger.warning("Volume category calculation expects a DataFrame with price and volume data. Returning 'Neutral Volume'.")
        return "Neutral Volume"
    if df.empty or 'volume' not in df.columns or 'open' not in df.columns or 'close' not in df.columns:
        logger.warning("Empty or invalid dataframe for volume category calculation. Returning 'Neutral Volume'.")
        return "Neutral Volume"
    try:
        vol_sma = df['volume'].rolling(window=sma_period).mean()
        bull_candle = df['close'] > df['open']
        if df['volume'].iloc[-1] > vol_sma.iloc[-1] * high_thresh:
            result = "High Bull Volume" if bull_candle.iloc[-1] else "High Bear Volume"
        elif df['volume'].iloc[-1] < vol_sma.iloc[-1] * low_thresh:
            result = "Low Bull Volume" if bull_candle.iloc[-1] else "Low Bear Volume"
        else:
            result = "Neutral Volume"
        logger.debug(f"Volume category: {result}")
        return result
    except Exception as e:
        logger.error(f"Error calculating volume weighted category: {e}")
        return "Neutral Volume"


def _pattern_signal(values: np.ndarray) -> np.ndarray:
    """Collapse TA-Lib style outputs (+/-100) into -1/0/1 signals."""

    return np.sign(np.asarray(values, dtype=float)).astype(int)


def get_candlestick_patterns(df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """Return the canonical candlestick pattern columns with ``suffix``.

    The pattern set mirrors the model feature schema and preserves direction by
    emitting ``-1`` for bearish readings, ``1`` for bullish readings and ``0``
    otherwise.
    """

    column_names = [f"{column}{suffix}" for column in DEFAULT_CANDLESTICK_PATTERN_COLUMNS]
    if df is None or df.empty:
        if df is None:
            logger.debug("Candlestick pattern request received empty dataframe.")
        index = getattr(df, "index", pd.Index([]))
        return pd.DataFrame(columns=column_names, index=index)

    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        logger.warning(
            "Dataframe missing required OHLC columns for candlestick patterns."
        )
        return pd.DataFrame(0, index=df.index, columns=column_names)

    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    patterns = pd.DataFrame(index=df.index)
    for display_name, feature_name, pattern_code in DEFAULT_CANDLESTICK_PATTERN_SPECS:
        func = getattr(talib, pattern_code, None)
        if func is None:
            logger.warning("Pattern function missing for %s", pattern_code)
            patterns[f"{feature_name}{suffix}"] = 0
            continue

        try:
            values = func(o, h, l, c)
            patterns[f"{feature_name}{suffix}"] = _pattern_signal(values)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Pattern %s failed: %s", pattern_code, exc)
            patterns[f"{feature_name}{suffix}"] = 0

    expected = [f"{col}{suffix}" for col in DEFAULT_CANDLESTICK_PATTERN_COLUMNS]
    return patterns.reindex(columns=expected, fill_value=0).astype(int)


# NEW: Bollinger Bands
def calculate_bollinger_bands(df, period=20, std_dev=2):
    """Calculate Bollinger Bands."""
    close_series = _resolve_close_series(df)
    if close_series is None or close_series.empty:
        logger.warning("Empty or invalid data for Bollinger Bands. Returning None.")
        return None, None, None
    try:
        middle = close_series.rolling(window=period).mean()
        std = close_series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        logger.debug(f"Bollinger Bands: Upper={upper.iloc[-1]}, Middle={middle.iloc[-1]}, Lower={lower.iloc[-1]}")
        return upper.iloc[-1], middle.iloc[-1], lower.iloc[-1]
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        return None, None, None


def calculate_adx(
    high: pd.DataFrame | pd.Series,
    low: pd.Series | None = None,
    close: pd.Series | None = None,
    period: int = 14,
) -> float | pd.Series | None:
    """Return the Average Directional Index (ADX) as a series or scalar."""

    def _ensure_series(values) -> pd.Series:
        if isinstance(values, pd.Series):
            return values.astype(float)
        return pd.Series(values, dtype=float)

    if isinstance(high, pd.DataFrame):
        df = high
        required_cols = {"high", "low", "close"}
        if df.empty or not required_cols.issubset(df.columns):
            logger.warning("Empty or invalid dataframe for ADX calculation. Returning None.")
            return None
        if len(df) < period + 1:
            logger.warning(
                "Not enough data points (%s) for ADX period %s. Returning None.",
                len(df),
                period,
            )
            return None
        high_series = df["high"].astype(float)
        low_series = df["low"].astype(float)
        close_series = df["close"].astype(float)
        return_series = False
    else:
        if low is None or close is None:
            raise ValueError("Low and close series are required when high is not a DataFrame")
        high_series = _ensure_series(high)
        low_series = _ensure_series(low)
        close_series = _ensure_series(close)
        return_series = True

    try:
        up_move = high_series.diff().fillna(0.0)
        down_move = (-low_series.diff()).fillna(0.0)

        plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

        tr_components = [
            (high_series - low_series),
            (high_series - close_series.shift(1)).abs(),
            (low_series - close_series.shift(1)).abs(),
        ]
        true_range = pd.concat(tr_components, axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().replace(0.0, np.nan)

        plus_dm_series = pd.Series(plus_dm, index=high_series.index)
        minus_dm_series = pd.Series(minus_dm, index=high_series.index)

        plus_di = 100.0 * (plus_dm_series.rolling(window=period).sum() / atr)
        minus_di = 100.0 * (minus_dm_series.rolling(window=period).sum() / atr)

        di_sum = (plus_di + minus_di).replace(0.0, np.nan)
        dx = (plus_di.subtract(minus_di).abs() / di_sum) * 100.0
        adx = dx.rolling(window=period).mean()

        if return_series:
            return adx

        last_adx = adx.iloc[-1] if not adx.empty else None
        if pd.isna(last_adx):
            return None
        logger.debug("ADX calculated: %s", last_adx)
        return float(last_adx)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Error calculating ADX: %s", exc)
        return None


def calculate_obv(
    close: pd.DataFrame | pd.Series,
    volume: pd.Series | None = None,
) -> float | pd.Series | None:
    """Return On-Balance Volume (OBV) as a series or scalar."""

    def _ensure_series(values) -> pd.Series:
        if isinstance(values, pd.Series):
            return values.astype(float)
        return pd.Series(values, dtype=float)

    if isinstance(close, pd.DataFrame):
        df = close
        required_cols = {"close", "volume"}
        if df.empty or not required_cols.issubset(df.columns):
            logger.warning("Empty or invalid dataframe for OBV calculation. Returning None.")
            return None
        close_series = df["close"].astype(float)
        volume_series = df["volume"].fillna(0.0).astype(float)
        return_series = False
    else:
        if volume is None:
            raise ValueError("Volume series is required when passing close as a Series")
        close_series = _ensure_series(close)
        volume_series = _ensure_series(volume).fillna(0.0)
        return_series = True

    try:
        price_diff = close_series.diff().fillna(0.0)
        direction = price_diff.apply(lambda val: 1 if val > 0 else (-1 if val < 0 else 0))
        obv_series = (direction * volume_series).cumsum()
        if return_series:
            return obv_series
        last_obv = obv_series.iloc[-1] if not obv_series.empty else None
        if pd.isna(last_obv):
            return None
        logger.debug("OBV calculated: %s", last_obv)
        return float(last_obv)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Error calculating OBV: %s", exc)
        return None


def calculate_stochastic_oscillator(
    high: pd.DataFrame | pd.Series,
    low: pd.Series | None = None,
    close: pd.Series | None = None,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series | float | None, pd.Series | float | None]:
    """Return Stochastic Oscillator %K and %D as series or scalars."""

    def _ensure_series(values) -> pd.Series:
        if isinstance(values, pd.Series):
            return values.astype(float)
        return pd.Series(values, dtype=float)

    def _empty_series(reference: pd.Series) -> pd.Series:
        return pd.Series(np.nan, index=reference.index)

    if isinstance(high, pd.DataFrame):
        df = high
        required_cols = {"high", "low", "close"}
        if df.empty or not required_cols.issubset(df.columns):
            logger.warning(
                "Empty or invalid dataframe for Stochastic calculation. Returning None."
            )
            return None, None
        if len(df) < k_period:
            logger.warning(
                "Not enough data points (%s) for Stochastic period %s. Returning None.",
                len(df),
                k_period,
            )
            return None, None
        high_series = df["high"].astype(float)
        low_series = df["low"].astype(float)
        close_series = df["close"].astype(float)
        return_series = False
    else:
        if low is None or close is None:
            raise ValueError("Low and close series are required when passing high as a Series")
        high_series = _ensure_series(high)
        low_series = _ensure_series(low)
        close_series = _ensure_series(close)
        return_series = True

    try:
        lowest_low = low_series.rolling(window=k_period).min()
        highest_high = high_series.rolling(window=k_period).max()
        range_ = (highest_high - lowest_low).replace(0.0, np.nan)

        percent_k = ((close_series - lowest_low) / range_) * 100.0
        percent_d = percent_k.rolling(window=d_period).mean()

        if return_series:
            return percent_k, percent_d

        last_k = percent_k.iloc[-1] if not percent_k.empty else None
        last_d = percent_d.iloc[-1] if not percent_d.empty else None

        last_k = None if pd.isna(last_k) else float(last_k)
        last_d = None if pd.isna(last_d) else float(last_d)

        logger.debug("Stochastic Oscillator calculated: %%K=%s, %%D=%s", last_k, last_d)
        return last_k, last_d
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Error calculating Stochastic Oscillator: %s", exc)
        if return_series:
            return _empty_series(high_series), _empty_series(high_series)
        return None, None


# =============================================================================
# BACKWARD-COMPATIBILITY SHIM FOR OLD MODELS (2024–2025 training data)
# =============================================================================
def add_legacy_candlestick_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert NEW generic candlestick schema → EXACT OLD 133-column schema
    that all current models + scaler were trained on.
    """
    df = df.copy()

    for tf in ["1h", "4h", "1d"]:
        # === RESTORE EXACT OLD COLUMN NAMES THAT MODELS EXPECT ===
        # 1. Engulfing
        engulf_col = f"pattern_engulfing_{tf}"
        if engulf_col in df.columns:
            s = df[engulf_col]
            # Some upstream joins can duplicate column names; guard against
            # DataFrame output so the legacy single-column assignment succeeds.
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s = s.fillna(0).astype(int)
            df[f"pattern_bullish_engulfing_{tf}"] = (s > 0).astype(int)
            df[f"pattern_bearish_engulfing_{tf}"] = (s < 0).astype(int)
        else:
            df[f"pattern_bullish_engulfing_{tf}"] = 0
            df[f"pattern_bearish_engulfing_{tf}"] = 0

        # 2. Marubozu (never existed)
        df[f"pattern_marubozu_bull_{tf}"] = 0
        df[f"pattern_marubozu_bear_{tf}"] = 0

        # 3. FORCE-ADD the old generic patterns that were renamed/dropped
        #     These must exist or models crash
        for pattern in ["hammer", "inverted_hammer", "shooting_star", "morning_star", "evening_star"]:
            old_col = f"pattern_{pattern}_{tf}"
            new_col = f"pattern_{pattern}_{tf}"
            if new_col in df.columns:
                df[old_col] = df[new_col].fillna(0).astype(int)
            else:
                df[old_col] = 0

    # === NOW safely drop only the truly new patterns (post-2025 refactor) ===
    truly_new_patterns = [
        "pattern_piercing_line_",
        "pattern_three_white_soldiers_",
        "pattern_hanging_man_",
        "pattern_three_black_crows_",
        "pattern_engulfing_",  # safe now — we already consumed it
    ]
    cols_to_drop = [col for col in df.columns if any(pat in col for pat in truly_new_patterns)]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    return df
