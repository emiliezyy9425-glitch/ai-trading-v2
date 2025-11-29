"""Utilities for writing CSV files with consistent timestamp handling."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import logging

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S+00:00"
TIMESTAMP_COLUMN = "timestamp"
MAX_INVALID_RATIO = 0.05


def _ensure_timestamp_series(df: pd.DataFrame, *, logger: logging.Logger) -> pd.DataFrame:
    """Return a copy of ``df`` with a normalized UTC timestamp column if present."""
    if TIMESTAMP_COLUMN not in df.columns:
        return df.copy()

    df_copy = df.copy()
    normalized = (
        df_copy[TIMESTAMP_COLUMN]
        .astype(str)
        .str.strip("\"'")
    )
    timestamps = pd.to_datetime(
        normalized,
        errors="coerce",
        utc=True,
        format="ISO8601",
    )
    invalid = int(timestamps.isna().sum())
    if invalid:
        logger.warning(
            "Detected %d timestamp values that could not be parsed; leaving as NaT.",
            invalid,
        )
    df_copy[TIMESTAMP_COLUMN] = timestamps
    return df_copy


def _format_timestamp_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with UTC timestamps formatted for CSV export."""

    if TIMESTAMP_COLUMN not in df.columns:
        return df

    formatted = df.copy()
    timestamps = formatted[TIMESTAMP_COLUMN]

    if not is_datetime64_any_dtype(timestamps.dtype):
        timestamps = pd.to_datetime(timestamps, errors="coerce", utc=True)
    elif is_datetime64tz_dtype(timestamps.dtype):
        timestamps = timestamps.dt.tz_convert("UTC")
    else:
        timestamps = timestamps.dt.tz_localize("UTC", errors="coerce")

    formatted[TIMESTAMP_COLUMN] = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    return formatted


def _boost_csv_field_size_limit(logger: logging.Logger) -> int | None:
    """Raise the csv field size limit to accommodate very wide rows.

    The pandas Python engine relies on the stdlib ``csv`` module which enforces a
    maximum field size. Large JSON blobs or encoded payloads can easily exceed the
    default ``131072`` character limit leading to ``_csv.Error`` exceptions.  We
    progressively attempt to raise the limit up to ``sys.maxsize`` while handling
    platforms where ``csv.field_size_limit`` cannot accept such a large value.
    """

    try:
        current_limit = csv.field_size_limit()
    except (TypeError, AttributeError):
        # Some exotic Python builds may not support querying the limit.  In that
        # case we cannot take corrective action, so return early.
        return None

    target = sys.maxsize
    if current_limit >= target:
        return current_limit

    while target > current_limit:
        try:
            new_limit = csv.field_size_limit(target)
        except (OverflowError, ValueError):
            target //= 2
            continue
        if new_limit != current_limit:
            logger.warning(
                "Increased csv.field_size_limit from %d to %d to accommodate wide rows.",
                current_limit,
                new_limit,
            )
        return new_limit

    logger.error(
        "Unable to increase csv.field_size_limit beyond %d; wide rows may still fail to parse.",
        current_limit,
    )
    return None


def save_dataframe_with_timestamp_validation(
    df: pd.DataFrame,
    path: str | Path,
    *,
    logger: Optional[logging.Logger] = None,
    validation_read_kwargs: Optional[Dict[str, Any]] = None,
    **to_csv_kwargs: Any,
) -> pd.DataFrame:
    """Persist ``df`` to *path* enforcing ISO8601 timestamps and reload validation."""
    if logger is None:
        logger = logging.getLogger(__name__)

    if "date_format" in to_csv_kwargs:
        raise ValueError("date_format cannot be overridden; timestamps must use ISO8601 format.")

    normalized = _ensure_timestamp_series(df, logger=logger)
    normalized = _format_timestamp_for_export(normalized)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    to_csv_kwargs.setdefault("index", False)
    to_csv_kwargs.setdefault("quoting", csv.QUOTE_MINIMAL)
    normalized.to_csv(destination, date_format=DEFAULT_DATE_FORMAT, **to_csv_kwargs)

    read_kwargs: Dict[str, Any] = {"low_memory": False}
    if validation_read_kwargs:
        read_kwargs.update(validation_read_kwargs)

    try:
        reloaded = pd.read_csv(destination, on_bad_lines="warn", **read_kwargs)
    except pd.errors.ParserError as exc:
        fallback_kwargs = dict(read_kwargs)
        fallback_kwargs["engine"] = "python"
        fallback_kwargs.pop("low_memory", None)
        fallback_kwargs.setdefault("on_bad_lines", "warn")
        _boost_csv_field_size_limit(logger)
        logger.warning(
            "ParserError when validating %s with C engine (%s); retrying with python engine.",
            destination,
            exc,
        )
        try:
            reloaded = pd.read_csv(destination, **fallback_kwargs)
        except pd.errors.ParserError as python_exc:
            fallback_kwargs = dict(fallback_kwargs)
            fallback_kwargs.pop("on_bad_lines", None)

            skipped_rows = []

            def _collect_bad_rows(row: list[str]) -> None:
                skipped_rows.append(row)

            fallback_kwargs["on_bad_lines"] = _collect_bad_rows
            logger.error(
                "Python engine also failed to parse %s (%s); skipping malformed rows.",
                destination,
                python_exc,
            )
            reloaded = pd.read_csv(destination, **fallback_kwargs)
            if skipped_rows:
                raise ValueError(
                    f"Encountered {len(skipped_rows)} malformed rows when validating {destination}."
                )
    if TIMESTAMP_COLUMN in reloaded.columns and len(reloaded):
        cleaned = (
            reloaded[TIMESTAMP_COLUMN]
            .astype(str)
            .str.strip("\"'")
        )
        timestamps = pd.to_datetime(
            cleaned,
            errors="coerce",
            utc=True,
            format="ISO8601",
        )
        invalid_after = int(timestamps.isna().sum())
        if invalid_after:
            ratio = invalid_after / len(reloaded)
            if ratio > MAX_INVALID_RATIO:
                raise ValueError(
                    f"{invalid_after} of {len(reloaded)} timestamps failed to parse when reloading {destination}"
                )
            logger.warning(
                "Detected %d/%d invalid timestamps when validating %s; downstream readers will drop them.",
                invalid_after,
                len(reloaded),
                destination,
            )
    return reloaded
