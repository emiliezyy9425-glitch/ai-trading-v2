"""Shared helpers for reading and caching the project's ticker universe."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from project_paths import resolve_data_path

logger = logging.getLogger(__name__)

TICKERS_FILE_PATH = resolve_data_path("tickers.txt")

_TICKERS_CACHE: list[str] | None = None
_TICKERS_MTIME: float | None = None


def _normalize_tickers(raw: list[str]) -> list[str]:
    return [ticker.strip().upper() for ticker in raw if ticker and ticker.strip()]


def load_tickers() -> list[str]:
    """Return the list of tickers from ``data/tickers.txt`` or ``["TSLA"]``."""

    path = Path(TICKERS_FILE_PATH)
    if path.exists():
        with path.open() as handle:
            tickers = _normalize_tickers(handle.readlines())
        if tickers:
            logger.info("✅ Loaded %d tickers from %s", len(tickers), path)
            return tickers
        logger.warning("⚠️ %s is empty. Using default ticker: TSLA", path)
    else:
        logger.warning("⚠️ %s not found. Using default ticker: TSLA", path)
    return ["TSLA"]


def get_cached_tickers(force_refresh: bool = False) -> list[str]:
    """Return cached tickers, reloading if the file changes or ``force_refresh``."""

    global _TICKERS_CACHE, _TICKERS_MTIME

    try:
        mtime = os.path.getmtime(TICKERS_FILE_PATH)
    except OSError:
        mtime = None

    if force_refresh or _TICKERS_CACHE is None or _TICKERS_MTIME != mtime:
        _TICKERS_CACHE = load_tickers()
        _TICKERS_MTIME = mtime
    return _TICKERS_CACHE


__all__ = ["TICKERS_FILE_PATH", "get_cached_tickers", "load_tickers"]
