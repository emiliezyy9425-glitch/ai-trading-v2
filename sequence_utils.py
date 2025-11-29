"""Utility helpers for preparing fixed-length ML sequences."""
from typing import Optional, List

import numpy as np
import pandas as pd


def pad_sequence_to_length(
    df: pd.DataFrame,
    target_length: int = 60,
    feature_columns: Optional[List[str]] = None,
) -> np.ndarray:
    """Return a (target_length, n_features) array suitable for Transformer/LSTM.

    - If df has < target_length rows  → left-pad with zeros (oldest timesteps = 0)
    - If df has exactly target_length → return as-is
    - If df has > target_length        → take the most recent target_length rows

    This matches exactly how the models were trained in self_learn.py.
    """
    if feature_columns is None:
        # Exclude non-feature columns like 'timestamp', 'ticker', etc.
        df_features = df.select_dtypes(include=[np.number])
    else:
        df_features = df[feature_columns]

    arr = df_features.values.astype(np.float32)

    if len(arr) >= target_length:
        arr = arr[-target_length:]
    else:
        pad_width = target_length - len(arr)
        arr = np.pad(arr, ((pad_width, 0), (0, 0)), mode="constant", constant_values=0.0)

    return arr
