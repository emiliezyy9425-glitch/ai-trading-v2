import logging
import csv
import os
from typing import Any

import numpy as np
import pandas as pd

from feature_engineering import (
    add_golden_price_features,
    count_fib_timezones,
    derive_fibonacci_features,
    encode_td9,
    encode_vol_cat,
    encode_zig,
)
from self_learn import FEATURE_NAMES  # Canonical model feature set (no raw prices)

from csv_utils import save_dataframe_with_timestamp_validation
from project_paths import resolve_data_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

import csv
csv.field_size_limit(10000000)

def parse_fibonacci(fib_value):
    """Parse fib levels robustly, returning six floats."""
    if not isinstance(fib_value, str) or not fib_value.strip():
        return [0.0] * 6
    try:
        levels = [
            float(v.split(":")[1].strip())
            for v in fib_value.split(",")
            if ":" in v
        ]
        return levels[:6] + [0.0] * (6 - len(levels))
    except Exception as e:
        logger.warning(f"Fib parse failed: {e}. Using defaults.")
        return [0.0] * 6


def _parse_timestamp_column(series: pd.Series) -> pd.Series:
    """Parse a timestamp column with multiple fallbacks.

    The historical data occasionally mixes ISO8601 strings and
    Unix epoch integers (in seconds or milliseconds). We coerce everything
    to pandas ``datetime64[ns, UTC]`` and raise a descriptive error if any
    rows remain invalid.
    """

    def _coerce_value(value: object):
        raw = str(value).strip().strip("\"'")
        if not raw or raw.lower() in {"nan", "nat"}:
            return pd.NaT

        # Primary attempt: rely on pandas/dateutil parsing.
        ts = pd.to_datetime(raw, utc=True, errors="coerce")
        if pd.notna(ts):
            return ts

        # Fallback for integer/float epoch values.
        numeric = pd.to_numeric(raw, errors="coerce")
        if pd.notna(numeric):
            for unit in ("s", "ms"):
                ts = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
                if pd.notna(ts):
                    return ts

        return pd.NaT

    parsed = series.apply(_coerce_value)

    if parsed.isna().any():
        missing_mask = parsed.isna()
        bad_rows = series.index[missing_mask].tolist()
        sample_values = series[missing_mask].astype(str).head(5).tolist()
        logger.warning(
            "Unable to parse timestamp values at rows: %s (sample values: %s). "
            "These rows will be dropped.",
            bad_rows[:5],
            sample_values,
        )

    return parsed


def _load_historical_data(path: str) -> pd.DataFrame:
    """Load *path* with multiple quoting strategies before falling back to python."""

    read_attempts = [
        {"quoting": csv.QUOTE_ALL, "escapechar": "\\"},
        {"quoting": csv.QUOTE_ALL},
        {"escapechar": "\\"},
        {},
    ]

    base_kwargs = {"low_memory": False}

    last_error: Exception | None = None
    for attempt in read_attempts:
        try:
            return pd.read_csv(path, **base_kwargs, **attempt)
        except Exception as exc:  # pragma: no cover - defensive logging
            last_error = exc
            logger.warning(
                "Failed to read %s with options %s: %s",
                path,
                attempt,
                exc,
            )

    try:
        logger.warning(
            "Falling back to python engine to load %s after C engine failures.",
            path,
        )
        return pd.read_csv(path, engine="python")
    except Exception as exc:
        if last_error is None:
            last_error = exc
        raise RuntimeError(f"Unable to load historical data from {path}: {last_error}") from exc


def add_decision_column(
    historical_path: str | os.PathLike[str] | None = None,
):
    if historical_path is None:
        historical_path = resolve_data_path("historical_data.csv")
    """Add a binary ``decision`` column (0=Buy, 1=Sell) to ``historical_data.csv``."""
    try:
        df = _load_historical_data(historical_path)
        logger.info(f"Loaded {len(df)} rows from {historical_path}")

        # Remove pandas-suffixed duplicate columns like 'foo.1', 'bar.2'
        suffixed = [c for c in df.columns if c.split('.')[-1].isdigit()]
        if suffixed:
            df = df.drop(columns=suffixed)
            logger.info(f"Removed suffixed duplicate columns: {suffixed}")

        # Drop any duplicated columns to prevent issues from repeated runs
        df = df.loc[:, ~df.columns.duplicated()].copy()
        logger.info(f"Dropped duplicated columns if any. Now {len(df.columns)} columns.")

        # Normalise timeframe suffixes so downstream features align with FEATURE_NAMES
        suffix_map = {"1hour": "1h", "4hours": "4h", "1day": "1d"}
        rename_map: dict[str, str] = {}
        for col in df.columns:
            for old_suffix, new_suffix in suffix_map.items():
                if col.endswith(old_suffix):
                    base = col[: -len(old_suffix)]
                    rename_map[col] = f"{base}{new_suffix}"
                    break

        if rename_map:
            df = df.rename(columns=rename_map)
            logger.info(
                "Renamed timeframe columns to canonical suffixes: %s",
                rename_map,
            )

        # Derive additional engineered features expected by training workflows
        def _normalise_fib_summary(value: Any) -> Any:
            if isinstance(value, dict):
                return ",".join(f"{k}:{v}" for k, v in value.items())
            return value

        for tf in ['1h', '4h', '1d']:
            td9_col = f'td9_summary_{tf}'
            if td9_col in df.columns and f'td9_{tf}' not in df.columns:
                df[f'td9_{tf}'] = df[td9_col].apply(encode_td9)

            zig_col = f'zig_zag_trend_{tf}'
            if zig_col in df.columns and f'zig_{tf}' not in df.columns:
                df[f'zig_{tf}'] = df[zig_col].apply(encode_zig)

            vol_cat_col = f'vol_category_{tf}'
            if vol_cat_col in df.columns and f'vol_cat_{tf}' not in df.columns:
                df[f'vol_cat_{tf}'] = df[vol_cat_col].apply(encode_vol_cat)

            tz_col = f'fib_time_zones_{tf}'
            if tz_col in df.columns and f'fib_time_count_{tf}' not in df.columns:
                df[f'fib_time_count_{tf}'] = df[tz_col].apply(count_fib_timezones)

            fib_col = f'fib_summary_{tf}'
            level1_col = f'fib_level1_{tf}'
            price_col = f'price_{tf}'
            if fib_col in df.columns:
                df[fib_col] = df[fib_col].apply(_normalise_fib_summary)

            if fib_col in df.columns and level1_col not in df.columns:
                prices = (
                    df[price_col]
                    if price_col in df.columns
                    else df.get('price', pd.Series([0.0] * len(df), index=df.index))
                )
                parsed_levels = df[fib_col].apply(parse_fibonacci)
                for i in range(6):
                    df[f'fib_level{i+1}_{tf}'] = parsed_levels.apply(lambda x: x[i])

                zone_deltas = [
                    derive_fibonacci_features(summary, price)[1]
                    for summary, price in zip(df[fib_col], prices)
                ]
                df[f'fib_zone_delta_{tf}'] = zone_deltas

                df.drop(columns=[fib_col], inplace=True)
                logger.info(f"Parsed {fib_col} into Fibonacci features.")

            # Drop auxiliary columns once encoded to avoid duplication downstream
            for extra_col in [td9_col, zig_col, vol_cat_col, tz_col]:
                if extra_col in df.columns and extra_col not in FEATURE_NAMES:
                    df.drop(columns=[extra_col], inplace=True)

        # Sort by timestamp and ticker
        if 'timestamp' in df.columns:
            df['timestamp'] = _parse_timestamp_column(df['timestamp'])
            invalid_mask = df['timestamp'].isna()
            if invalid_mask.any():
                drop_count = invalid_mask.sum()
                logger.warning(
                    "Dropping %d rows with invalid timestamps after parsing.",
                    drop_count,
                )
                df = df.loc[~invalid_mask].copy()
            if 'ticker' in df.columns:
                df = df.sort_values(['ticker', 'timestamp']).reset_index(drop=True)
            else:
                df = df.sort_values('timestamp').reset_index(drop=True)

        # Add golden price derivatives then drop raw price columns
        if 'timestamp' in df.columns:
            df = add_golden_price_features(df)


        # Always recalculate decision, forcing Buy/Sell only (no Hold)
        logger.info("Recalculating decision column, forcing Buy/Sell only.")
        if 'price_1h' in df.columns:
            df['future_price'] = df['price_1h'].shift(-8)
            df['future_return'] = (df['future_price'] - df['price_1h']) / df['price_1h']
        else:
            logger.warning("No 'price_1h' column. Using dummy (all HOLD).")
            df['future_return'] = 0.0

        # Compute decisions only where future_return is defined
        notna_mask = df['future_return'].notna()
        if not notna_mask.any():
            logger.warning(
                "Not enough rows to compute future returns. Keeping existing decisions."
            )
            if 'decision' not in df.columns:
                df['decision'] = 0  # default Buy if missing
        else:
            df = df.loc[notna_mask]
            df['decision'] = np.where(df['future_return'] >= 0, 0, 1)  # 0=Buy, 1=Sell
            logger.info(
                f"Label distribution after forcing Buy/Sell: \n{df['decision'].value_counts()}"
            )

        df = df.drop(columns=['future_price', 'future_return'], errors='ignore')

        # Drop any remaining raw price columns now that decisions are computed
        dropped_prices = [
            col
            for col in ['price_1h', 'price_4h', 'price_1d']
            if col in df.columns
        ]
        if dropped_prices:
            df = df.drop(columns=dropped_prices)
            logger.info("Removed raw price columns after labeling: %s", dropped_prices)

        # Align to FEATURE_NAMES (86)
        core_cols = ['timestamp', 'ticker', 'decision']
        price_cols = [col for col in df.columns if col.startswith('price_') and col not in FEATURE_NAMES]  # Raw prices only
        feature_cols = [c for c in df.columns if c not in core_cols + price_cols]
        missing = [c for c in FEATURE_NAMES if c not in feature_cols]
        if missing:
            logger.warning(f"Adding missing features: {missing}")
            for col in missing:
                df[col] = 0.0
        extra = [c for c in feature_cols if c not in FEATURE_NAMES]
        if extra:
            logger.warning(f"Dropping unexpected columns: {extra}")
            df = df.drop(columns=extra)

        # Assemble final dataframe and save
        final_df = df[core_cols[:2] + price_cols + FEATURE_NAMES + [core_cols[2]]]
        save_dataframe_with_timestamp_validation(
            final_df,
            historical_path,
            quoting=csv.QUOTE_ALL,
            logger=logger,
        )
        logger.info(
            f"Saved historical data (shared for supervised and PPO) to {historical_path} with {len(final_df)} rows"
        )

        # Validate feature count
        expected_features = len(FEATURE_NAMES)  # 86
        actual_features = len(final_df[FEATURE_NAMES].columns)
        if actual_features != expected_features:
            logger.error(
                f"Feature count mismatch in data: expected {expected_features}, got {actual_features}"
            )
            raise ValueError(
                f"Expected {expected_features} features, got {actual_features}"
            )

        return final_df

    except Exception as exc:
        logger.error(f"Failed to add decision column: {exc}")
        raise


if __name__ == "__main__":
    add_decision_column()