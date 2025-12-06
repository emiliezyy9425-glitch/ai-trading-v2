#!/usr/bin/env python3
"""
run_training_pipeline.py
Full end-to-end training pipeline:
1. Regenerate historical data (if needed)
2. Clean & validate
3. Add decision column
4. Create price-free dataset
5. Train all models
"""
import os
import csv
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

# ------------------------------------------------------------------
#  Fix: Use local PROJECT_ROOT instead of importing from project_paths
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # /app/scripts → /app

# ------------------------------------------------------------------
#  Bootstrap: Ensure project root is on Python path
# ------------------------------------------------------------------
try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

# ------------------------------------------------------------------
#  Imports
# ------------------------------------------------------------------
from clean_historical_data import clean_historical_data
from add_decision_column import add_decision_column
from csv_utils import save_dataframe_with_timestamp_validation
from project_paths import get_data_dir, resolve_data_path

# ------------------------------------------------------------------
#  Logging
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MIN_REQUIRED_ROWS = 100


# ------------------------------------------------------------------
#  Dataset loading
# ------------------------------------------------------------------
def _load_existing_dataset(path: Path) -> Optional[pd.DataFrame]:
    """Attempt to load CSV with multiple quoting strategies."""
    read_attempts = [
        {"quoting": csv.QUOTE_ALL, "escapechar": "\\"},
        {"escapechar": "\\"},
        {},
    ]
    base_kwargs = {"low_memory": False}

    last_error: Optional[Exception] = None
    for kwargs in read_attempts:
        try:
            df = pd.read_csv(path, **base_kwargs, **kwargs)
            return df
        except Exception as exc:
            last_error = exc
            logger.warning("Failed to read %s with options %s: %s", path, kwargs, exc)
    if last_error:
        logger.error("Unable to read %s: %s", path, last_error)
    return None


def _dataset_is_viable(df: pd.DataFrame) -> bool:
    """Check row count and label balance."""
    if len(df) < MIN_REQUIRED_ROWS:
        logger.warning("Only %d rows; need >= %d", len(df), MIN_REQUIRED_ROWS)
        return False

    if 'decision' in df.columns:
        unique = df['decision'].nunique()
        if unique < 2:
            logger.warning("Only %d unique decision(s); need >= 2", unique)
            return False
    return True


# ------------------------------------------------------------------
#  Regeneration placeholder
# ------------------------------------------------------------------
def _regenerate_historical_dataset(hist_file: Path) -> bool:
    """Regenerate historical_data.csv using your data pipeline."""
    try:
        # Replace with your actual data generation script
        from scripts.generate_historical_data import generate_historical_data
        success = generate_historical_data(str(hist_file))
        if success:
            logger.info("Regenerated historical dataset: %s", hist_file)
        return success
    except Exception as e:
        logger.error("Failed to regenerate data: %s", e)
        return False


# ------------------------------------------------------------------
#  Main pipeline
# ------------------------------------------------------------------
def main():
    logger.info("Starting training pipeline...")

    # Step 1: Resolve paths
    hist_file = resolve_data_path('historical_data.csv')
    supervised_file = resolve_data_path('historical_data_no_price.csv')

    # Step 2: Load or regenerate historical data
    df = _load_existing_dataset(hist_file)
    needs_regeneration = df is None or not _dataset_is_viable(df)

    if needs_regeneration:
        logger.info("Historical dataset not found or invalid; regenerating...")
        if not _regenerate_historical_dataset(hist_file):
            logger.error("Failed to regenerate dataset. Aborting.")
            return

    # Step 2.5: Clean & validate
    if not clean_historical_data(str(hist_file), str(hist_file)):
        logger.error("Data failed cleaning. Aborting.")
        return

    # Step 2.6: Add decision column
    add_decision_column(hist_file)
    logger.info("Added decision column.")

    # Step 2.7: Create price-free dataset
    df_supervised = pd.read_csv(hist_file)
    price_cols = [c for c in ['price', 'price_1h', 'price_4h', 'price_1d'] if c in df_supervised.columns]
    if price_cols:
        df_supervised = df_supervised.drop(columns=price_cols)

    save_dataframe_with_timestamp_validation(
        df_supervised, supervised_file, quoting=csv.QUOTE_MINIMAL, logger=logger
    )
    df_supervised.to_csv(supervised_file, index=False, quoting=csv.QUOTE_MINIMAL)
    logger.info(f"Saved price-free dataset: {supervised_file}")

    # Step 3: Train all models
    logger.info("Training models...")
    config_to_type = [
        ('rf_config.json', 'RandomForest'),
        ('xgb_config.json', 'XGBoost'),
        ('lgb_config.json', 'LightGBM'),
        ('ppo_config.json', 'PPO'),
        ('transformer_config.json', 'Transformer'),
        ('lstm_config.json', 'LSTM'),
    ]

    from retrain_models import train_model

    for config_file, model_type in config_to_type:
        config_path = PROJECT_ROOT / config_file
        if config_path.exists():
            logger.info(f"Training {model_type} → {config_path}")
            train_model(model_type, str(config_path))
        else:
            logger.warning(f"Config not found: {config_path} → skipping {model_type}")

    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()