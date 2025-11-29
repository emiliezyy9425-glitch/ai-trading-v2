"""Shared helpers for resolving the training lookback horizon.

All historical-data generation workflows should agree on how many years of
market data to pull so training and live features stay aligned.  The helper
functions in this module centralise the "10 years by default" policy while
allowing overrides through the ``TRAINING_LOOKBACK_YEARS`` environment variable.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

DEFAULT_LOOKBACK_YEARS = 10
_ENV_VAR = "TRAINING_LOOKBACK_YEARS"

logger = logging.getLogger(__name__)


def _parse_env_value(value: str | None) -> int | None:
    if not value:
        return None
    try:
        years = int(value)
    except ValueError:
        logger.warning(
            "Invalid %s value '%s'; falling back to %d years.",
            _ENV_VAR,
            value,
            DEFAULT_LOOKBACK_YEARS,
        )
        return None
    if years <= 0:
        logger.warning(
            "%s must be positive; falling back to %d years.",
            _ENV_VAR,
            DEFAULT_LOOKBACK_YEARS,
        )
        return None
    return years


@lru_cache(maxsize=1)
def get_training_lookback_years() -> int:
    """Return the number of years of history to generate (default ``10``)."""

    env_years = _parse_env_value(os.getenv(_ENV_VAR))
    years = env_years if env_years is not None else DEFAULT_LOOKBACK_YEARS
    if years != DEFAULT_LOOKBACK_YEARS:
        logger.info("Using %d years of historical data for training.", years)
    return years


def get_training_lookback_days() -> int:
    """Return the lookback horizon expressed in days."""

    return get_training_lookback_years() * 365


def get_training_lookback_duration_string() -> str:
    """Return an IBKR duration string such as ``"10 Y"``."""

    return f"{get_training_lookback_years()} Y"
