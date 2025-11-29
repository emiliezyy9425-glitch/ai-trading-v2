import logging
import os
import shutil
import csv

import pandas as pd

from csv_utils import save_dataframe_with_timestamp_validation
from project_paths import resolve_data_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

import csv
csv.field_size_limit(10000000)

def clean_historical_data(input_file, output_file):
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} does not exist.")
        return False

    backup_file = f"{input_file}.bak"
    file_size = os.path.getsize(input_file)

    if file_size == 0:
        if os.path.exists(backup_file) and os.path.getsize(backup_file) > 0:
            try:
                shutil.copyfile(backup_file, input_file)
                file_size = os.path.getsize(input_file)
                logger.warning(
                    "Input file %s was empty. Restored contents from backup %s before cleaning.",
                    input_file,
                    backup_file,
                )
            except OSError as exc:
                logger.error(
                    "Failed to restore %s from backup %s: %s", input_file, backup_file, exc
                )
                return False
        else:
            logger.error(
                "Historical data file %s is empty and no valid backup is available.",
                input_file,
            )
            return False

    backup_dir = os.path.dirname(os.path.abspath(backup_file)) or "."
    if os.access(backup_dir, os.W_OK):
        if file_size > 0:
            try:
                shutil.copyfile(input_file, backup_file)
                logger.info(f"Backed up {input_file} to {backup_file}")
            except PermissionError:
                logger.warning(
                    "Insufficient permissions to create backup %s. Continuing without backup.",
                    backup_file,
                )
            except OSError as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Unable to create backup %s due to %s. Continuing without backup.",
                    backup_file,
                    exc,
                )
    else:
        logger.warning(
            "Directory %s is not writable. Skipping creation of backup file %s.",
            backup_dir,
            backup_file,
        )

    read_attempts = [
        {"quoting": csv.QUOTE_ALL, "escapechar": "\\"},
        {"quoting": csv.QUOTE_ALL},
        {"escapechar": "\\"},
        {},
    ]

    df = None
    last_error = None
    for attempt in read_attempts:
        try:
            df = pd.read_csv(
                input_file,
                low_memory=False,
                **attempt,
            )
            break
        except pd.errors.EmptyDataError:
            logger.error(
                "Historical data file %s is empty."
                " Restore it from backup or regenerate the dataset before retrying.",
                input_file,
            )
            return False
        except Exception as exc:  # pragma: no cover - defensive logging
            last_error = exc
            logger.warning(
                "Failed to read %s with options %s: %s",
                input_file,
                attempt,
                exc,
            )

    if df is None:
        logger.error(
            "Unable to load historical data from %s. Last error: %s",
            input_file,
            last_error,
        )
        return False

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    logger.info(f"Loaded {len(df)} rows from {input_file}")

    bool_cols = [col for col in df.columns if "high_vol" in col or "vol_spike" in col]
    string_cols = [
        col
        for col in df.columns
        if "fib_summary" in col
        or "fib_time_zones" in col
        or "pivot_points" in col
        or "zig_zag_trend" in col
        or "vol_category" in col
        or "td9_summary" in col
    ]

    numeric_cols = []
    for col in df.columns:
        if col == "timestamp":
            continue
        if col == "ticker":
            df[col] = df[col].astype(str).fillna("")
            continue
        if col in bool_cols or col in string_cols:
            continue
        numeric_cols.append(col)

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in string_cols:
        default = (
            "Neutral"
            if any(token in col for token in ["trend", "category", "summary"])
            else "0:0"
        )
        df[col] = (
            df[col]
            .astype(str)
            .replace({"nan": default, "NaN": default, "": default})
        )

    for col in bool_cols:
        series = df[col]
        df[col] = (
            series.astype(str)
            .str.strip()
            .str.lower()
            .replace({
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            })
        )
        df[col] = df[col].replace("nan", False)
        df[col] = df[col].fillna(False).astype(bool)

    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

    missing_counts = df.isna().sum()
    total_missing = int(missing_counts.sum())
    if total_missing:
        all_rows = len(df)
        mostly_missing = missing_counts[missing_counts >= int(0.9 * all_rows)]
        if not mostly_missing.empty:
            dropped_cols = mostly_missing.index.tolist()
            df = df.drop(columns=dropped_cols)
            logger.warning(
                "Dropped %d columns with >=90%% missing values: %s",
                len(dropped_cols),
                ", ".join(dropped_cols[:10]) + ("..." if len(dropped_cols) > 10 else ""),
            )
            missing_counts = df.isna().sum()
            total_missing = int(missing_counts.sum())

    if total_missing:
        logger.warning(
            "%d missing values remain after initial cleaning. Filling with defaults.",
            total_missing,
        )
        numeric_cols_after = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols_after) > 0:
            df[numeric_cols_after] = df[numeric_cols_after].fillna(0)
        object_cols_after = df.select_dtypes(include=["object"]).columns
        for col in object_cols_after:
            df[col] = df[col].fillna(
                "Neutral"
                if any(token in col for token in ["trend", "category", "summary"])
                else "Unknown"
            )
        bool_cols_after = df.select_dtypes(include=["bool"]).columns
        for col in bool_cols_after:
            df[col] = df[col].fillna(False)

    remaining_missing = int(df.isna().sum().sum())
    if remaining_missing:
        logger.warning("%d missing values could not be resolved.", remaining_missing)
    else:
        logger.info("No missing values after cleaning.")

    save_dataframe_with_timestamp_validation(
        df,
        output_file,
        quoting=csv.QUOTE_ALL,
        logger=logger,
    )
    logger.info(f"Cleaned data saved to {output_file}")
    return True


if __name__ == "__main__":
    hist_path = resolve_data_path("historical_data.csv")
    clean_historical_data(str(hist_path), str(hist_path))
