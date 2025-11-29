"""Generate supervised labels for historical data.

This script recalculates the ``decision`` column (0=Buy, 1=Sell) using the
canonical features available in ``historical_data.csv``. It preserves the
golden price features and the canonical ``FEATURE_NAMES`` to keep the dataset
compatible with the training pipelines.
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Iterable

import numpy as np
import pandas as pd

from csv_utils import save_dataframe_with_timestamp_validation
from project_paths import resolve_data_path
from self_learn import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Price-derived features that must be retained alongside the canonical features
GOLDEN_PRICE_FEATURES: set[str] = {"ret_1h", "ret_4h", "ret_24h", "price_z_120h", "bb_position_1h"}


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Ensure *columns* exist in *df*, filling missing ones with zeros."""

    for col in columns:
        if col not in df.columns:
            df[col] = 0.0
    return df


def add_decision_column(historical_path: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    """Add a binary ``decision`` column to ``historical_data.csv``.

    The label is computed as the summed 8-hour forward return using the ``ret_1h``
    feature. Any unexpected columns are dropped, while golden price features and
    canonical ``FEATURE_NAMES`` are retained. Missing canonical features are
    filled with zeros to maintain a stable schema.
    """

    path = resolve_data_path("historical_data.csv") if historical_path is None else historical_path
    logger.info("Loading %s", path)
    df = pd.read_csv(path, parse_dates=["timestamp"])

    if "ret_1h" not in df.columns:
        raise ValueError("ret_1h missing — cannot create labels")

    df["future_return_8h"] = df["ret_1h"].shift(-8).rolling(8).sum()
    df["decision"] = np.where(df["future_return_8h"] >= 0, 0, 1)  # 0=Buy, 1=Sell

    buy_pct = (df["decision"] == 0).mean()
    logger.info("REAL LABELS CREATED → %.1f%% Buy / %.1f%% Sell", buy_pct * 100, (1 - buy_pct) * 100)
    df = df.drop(columns=["future_return_8h"], errors="ignore")

    core = ["timestamp", "ticker", "decision"]
    valid_columns = set(core) | GOLDEN_PRICE_FEATURES | set(FEATURE_NAMES)
    foreign = set(df.columns) - valid_columns
    if foreign:
        logger.info("Dropping foreign columns: %s", sorted(foreign))
        df = df.drop(columns=foreign)

    df = _ensure_columns(df, FEATURE_NAMES)

    final_df = df[
        ["timestamp", "ticker"]
        + [col for col in GOLDEN_PRICE_FEATURES if col in df.columns]
        + sorted(FEATURE_NAMES)
        + ["decision"]
    ]

    save_dataframe_with_timestamp_validation(final_df, path, quoting=csv.QUOTE_ALL)
    logger.info("SUCCESS: %d rows with real labels saved", len(final_df))
    return final_df


if __name__ == "__main__":
    add_decision_column()
