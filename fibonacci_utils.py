"""Utilities for parsing serialized Fibonacci level summaries."""
from __future__ import annotations

import math
import re
from typing import Iterable, Iterator

_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_STRIP_CHARS = "{}[]'\" "


def _coerce_numeric(value: object) -> float | None:
    """Return ``value`` as a float when it represents a finite number."""

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        if math.isnan(float(value)) or not math.isfinite(float(value)):
            return None
        return float(value)

    if isinstance(value, str):
        match = _NUMERIC_RE.search(value)
        if match:
            try:
                number = float(match.group())
            except ValueError:
                return None
            if math.isnan(number) or not math.isfinite(number):
                return None
            return number
    return None


def _iterate_values(serialised: object) -> Iterator[object]:
    """Yield candidate numeric values from ``serialised``."""

    if isinstance(serialised, dict):
        yield from serialised.values()
        return

    if isinstance(serialised, str):
        text = serialised.strip()
        if not text:
            return
        cleaned = text.translate({ord(ch): " " for ch in _STRIP_CHARS})
        for part in cleaned.split(","):
            if ":" not in part:
                continue
            _, raw_value = part.split(":", 1)
            candidate = raw_value.strip()
            if candidate:
                yield candidate
        return

    if isinstance(serialised, Iterable):
        yield from serialised
        return


def parse_fibonacci_levels(value: object, max_levels: int = 6) -> list[float]:
    """Extract up to ``max_levels`` Fibonacci price levels from ``value``.

    Parameters
    ----------
    value:
        Serialized representation of Fibonacci levels. Supported formats include
        dictionaries (already parsed), comma-separated summary strings such as
        ``"0.382: 12.3, 0.5: Neutral"`` and general iterables.
    max_levels:
        The maximum number of levels to return. Missing values are padded with
        ``0.0``.
    """

    levels: list[float] = []
    for candidate in _iterate_values(value):
        numeric = _coerce_numeric(candidate)
        if numeric is None:
            continue
        levels.append(numeric)
        if len(levels) >= max_levels:
            break

    if len(levels) < max_levels:
        levels.extend([0.0] * (max_levels - len(levels)))
    else:
        levels = levels[:max_levels]
    return levels


def normalize_fibonacci_levels(value: object, price: object, max_levels: int = 5) -> list[float]:
    """Return Fibonacci levels normalized by ``price``.

    Non-finite price inputs (including zero) result in an all-zero response to
    avoid division errors.
    """

    levels = parse_fibonacci_levels(value, max_levels=max_levels)
    price_numeric = _coerce_numeric(price)
    if price_numeric in (None, 0.0):
        return [0.0] * max_levels
    return [level / price_numeric for level in levels]


__all__ = ["parse_fibonacci_levels", "normalize_fibonacci_levels"]
