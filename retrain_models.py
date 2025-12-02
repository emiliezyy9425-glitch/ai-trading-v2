# retrain_models.py (updated with fixes: scoring='roc_auc', XGBoost GPU params, gymnasium import, Transformer num_heads=1, LSTM save to .pt, removed SMOTE)
import json
import argparse
import logging
import os
import csv
import tempfile
import datetime
from itertools import chain
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from joblib import dump
import joblib

from self_learn import FEATURE_NAMES  # ← ONLY THIS
from project_paths import resolve_data_path
from fibonacci_utils import parse_fibonacci_levels
from utils.closed_bar_shift import apply_closed_bar_shift

import optuna
from optuna.trial import Trial
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer import TransformerModel
from lstm import AttentiveBiLSTM


from pytorch_utils import configure_pytorch

import gymnasium as gym
from ppo import TradingEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv


from feature_engineering import create_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset

from self_learn import FEATURE_NAMES






def make_env(df_subset: pd.DataFrame):
    return lambda: TradingEnv(data=df_subset.copy())

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_DIR / "retrain_models.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def _model_directories() -> List[Path]:
    """Return all directories that should receive trained model artifacts."""
    seen = []
    for candidate in [Path("/app/models"), MODEL_DIR]:
        if candidate not in seen:
            seen.append(candidate)
    return seen


def _persist_lstm_artifacts(model_state: Dict[str, torch.Tensor], config: Dict[str, Any]) -> None:
    """Save LSTM weights + config to Docker volume and repo copy when available."""
    saved_weight_paths: List[Path] = []
    saved_config_paths: List[Path] = []

    for directory in _model_directories():
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Unable to create model directory %s: %s", directory, exc)
            continue

        weights_path = directory / "updated_lstm.pt"
        torch.save(model_state, str(weights_path))
        saved_weight_paths.append(weights_path)

        cfg_path = directory / "updated_lstm.json"
        with open(cfg_path, "w") as cfg_file:
            json.dump(config, cfg_file, indent=2)
        saved_config_paths.append(cfg_path)

    if not saved_weight_paths or not saved_config_paths:
        logger.error("Failed to persist LSTM artifacts to any model directory")
    else:
        logger.info("Saved LSTM weights → %s", ", ".join(map(str, saved_weight_paths)))
        logger.info("Saved LSTM config → %s", ", ".join(map(str, saved_config_paths)))

configure_pytorch(logger)

MIN_SUPERVISED_ROWS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
# --------------------------------------------------------------------------- #
# Load Data
# --------------------------------------------------------------------------- #
def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load, label, and prepare training data."""
    df = pd.read_csv(historical_data.csv, parse_dates=["timestamp"])
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    df = apply_closed_bar_shift(df)

    # Regime filters / macro overlays should also only use closed data.
    if "sp500_above_20d" in df.columns:
        df["sp500_above_20d"] = df["sp500_above_20d"].shift(1)

    df = df.dropna(subset=FEATURE_NAMES, how="any")

    # Binary labeling: Buy (1) if positive return over the next 5 bars, else Sell (0)
    future_close = df.groupby("ticker")["close"].shift(-5)
    df["future_return"] = future_close / df["close"] - 1
    df = df.dropna(subset=["future_return"])
    df["label"] = (df["future_return"] > 0.01).astype(int)

    X = df[FEATURE_NAMES]
    y = df["label"]
    return X, y
# --------------------------------------------------------------------------- #
# New function to log training results
# --------------------------------------------------------------------------- #
def log_training_results(model_type: str, metrics: Dict[str, Any]):
    """Log training metrics to a JSONL file for historical tracking."""
    results_file = LOG_DIR / "model_results.jsonl"
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model_type,
        "metrics": metrics
    }
    with open(results_file, 'a') as f:
        f.write(json.dumps(entry) + '\n')
    logger.info(f"Logged training results for {model_type}: {metrics}")

# --------------------------------------------------------------------------- #
# Helper: resolve feature lists
# --------------------------------------------------------------------------- #
def _resolve_feature_lists(
    selected_features: Optional[List[str]],
) -> Tuple[List[str], List[str]]:
    if selected_features and not isinstance(selected_features, list):
        logger.warning("selected_features is not a list; using all features.")
        selected_features = None

    all_features = FEATURE_NAMES if 'FEATURE_NAMES' in globals() else []
    if selected_features:
        valid_features = [f for f in selected_features if f in all_features]
        if len(valid_features) < len(selected_features):
            logger.warning(
                "Some selected features not in FEATURE_NAMES; using valid ones: %s",
                valid_features,
            )
        features = valid_features
    else:
        features = all_features

    # Pattern features (subset of all)
    pattern_features = [f for f in features if f.startswith("pattern_")]

    return features, pattern_features


# --------------------------------------------------------------------------- #
# Data loading and preprocessing
# --------------------------------------------------------------------------- #
def load_and_preprocess_data(
    data_path: str,
    target_column: str,
    scale_features: bool = False,
    selected_features: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    List[str],
    np.ndarray,
    np.ndarray,
    Optional[StandardScaler],
]:
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path, low_memory=False)
    if len(df) < MIN_SUPERVISED_ROWS:
        logger.error("Insufficient data rows (%d < %d). Aborting training.", len(df), MIN_SUPERVISED_ROWS)
        raise ValueError("Insufficient data for training")

    features, _ = _resolve_feature_lists(selected_features)

    X = df[features]
    y = df[target_column]

    # Split into train + temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + 0.1, random_state=random_state, stratify=y
    )
    # Split temp into val + test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size / (test_size + 0.1), random_state=random_state, stratify=y_temp
    )

    scaler: Optional[StandardScaler] = None
    if scale_features:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=features, index=X_val.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)

    y_val_np = np.array(y_val)
    val_returns = np.array(df.loc[X_val.index, 'returns']) if 'returns' in df.columns else np.zeros(len(y_val))

    # New: Assert multi-class and consistent lengths
    if len(np.unique(y_val_np)) < 2:
        raise ValueError(f"Validation set has only {len(np.unique(y_val_np))} unique classes. Ensure stratified split works.")
    assert len(y_val_np) == len(val_returns), f"Mismatch: y_val {len(y_val_np)} vs val_returns {len(val_returns)}"

    logger.info("Data loaded: %d samples, %d features (train: %d, val: %d, test: %d)", len(df), len(features), len(X_train), len(X_val), len(X_test))

    return X_train, y_train, X_val, y_val, X_test, y_test, features, y_val_np, val_returns, scaler
# --------------------------------------------------------------------------- #
# Random Forest
# --------------------------------------------------------------------------- #
def train_random_forest(params: Dict[str, Any]):
    X_train, y_train, X_val, y_val, X_test, y_test, features, y_val_np, val_returns, _ = load_and_preprocess_data(
        data_path=params["data_path"],
        target_column=params["target_column"],
        scale_features=params.get("scale_features", False),
        selected_features=params.get("selected_features"),
        test_size=params.get("test_size", 0.2),
        random_state=params.get("random_state", 42),
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model_params = params.get("model_params", {})
    model = RandomForestClassifier(**model_params)

    model.fit(X_train_scaled, y_train)

    probs = model.predict_proba(X_val_scaled)[:, 1]
    preds = model.predict(X_val_scaled)

    auc = roc_auc_score(y_val, probs)
    acc = accuracy_score(y_val, preds)

    logger.info("RandomForest AUC: %.4f, Accuracy: %.4f", auc, acc)

    output_path = str(MODEL_DIR / "rf_latest.joblib")
    dump(model, output_path)
    logger.info("Saved RandomForest model → %s", output_path)

    joblib.dump(scaler, str(MODEL_DIR / "scaler.joblib"))
    joblib.dump(list(features), str(MODEL_DIR / "feature_order.joblib"))
    logger.info("Saved scaler.joblib and feature_order.joblib for RandomForest")

    

    # Log results
    metrics = {"AUC": auc, "Accuracy": acc}
    log_training_results("RandomForest", metrics)

# --------------------------------------------------------------------------- #
# XGBoost
# --------------------------------------------------------------------------- #
def train_xgboost(params: Dict[str, Any]):
    X_train, y_train, X_val, y_val, X_test, y_test, features, y_val_np, val_returns, _ = load_and_preprocess_data(
        data_path=params["data_path"],
        target_column=params["target_column"],
        scale_features=params.get("scale_features", False),
        selected_features=params.get("selected_features"),
        test_size=params.get("test_size", 0.2),
        random_state=params.get("random_state", 42),
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model_params = params.get("model_params", {})
    model_params.update({
        "tree_method": "hist" if torch.cuda.is_available() else "auto",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    })
    model = XGBClassifier(**model_params)

    model.fit(X_train_scaled, y_train)

    probs = model.predict_proba(X_val_scaled)[:, 1]
    preds = model.predict(X_val_scaled)

    auc = roc_auc_score(y_val, probs)
    acc = accuracy_score(y_val, preds)

    logger.info("XGBoost AUC: %.4f, Accuracy: %.4f", auc, acc)

    output_path = str(MODEL_DIR / "xgb_latest.joblib")
    dump(model, output_path)
    logger.info("Saved XGBoost model → %s", output_path)

    joblib.dump(scaler, str(MODEL_DIR / "scaler.joblib"))
    joblib.dump(list(features), str(MODEL_DIR / "feature_order.joblib"))
    logger.info("Saved scaler.joblib and feature_order.joblib for XGBoost")



    # Log results
    metrics = {"AUC": auc, "Accuracy": acc}
    log_training_results("XGBoost", metrics)
# --------------------------------------------------------------------------- #
# LightGBM
# --------------------------------------------------------------------------- #
def train_lightgbm(params: Dict[str, Any]):
    data_path = str(resolve_data_path(params.get("data_path", "historical_data.csv")))
    target_column = params.get("target_column", "signal")
    output_model_path = str(MODEL_DIR / params.get("output_model", "lgb_latest.joblib"))
    output_importance_path = str(MODEL_DIR / "lgb_feature_importance.joblib")
    scale_features = params.get("scale_features", False)
    selected_features = params.get("selected_features", None)
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)

    X_train, y_train, X_val, y_val, X_test, y_test, features, y_val_np, val_returns, _ = load_and_preprocess_data(
        data_path,
        target_column,
        scale_features,
        selected_features,
        test_size,
        random_state,
    )

    # Add assertion to check lengths
    assert len(X_train) == len(y_train), f"X_train and y_train length mismatch: {len(X_train)} vs {len(y_train)}"
    assert len(X_val) == len(y_val), f"X_val and y_val length mismatch: {len(X_val)} vs {len(y_val)}"
    assert len(X_test) == len(y_test), f"X_test and y_test length mismatch: {len(X_test)} vs {len(y_test)}"

    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        device = "gpu" if torch.cuda.is_available() else "cpu"
        model_params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "device": device,
            "random_state": random_state,
            "verbosity": -1,
        }

        try:
            model = LGBMClassifier(**model_params)
            cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='roc_auc')
            return cv_scores.mean()
        except Exception as e:
            if "OpenCL" in str(e) or "gpu" in str(e).lower():
                logger.warning("GPU training failed (%s). Falling back to CPU.", e)
                model_params["device"] = "cpu"
                model = LGBMClassifier(**model_params)
                cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='roc_auc')
                return cv_scores.mean()
            else:
                raise

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=params.get("n_trials", 50))

    best_params = study.best_params
    best_params.update({
        "random_state": random_state,
        "verbosity": -1,
    })

    device = "gpu" if torch.cuda.is_available() else "cpu"
    best_params["device"] = device

    try:
        model = LGBMClassifier(**best_params)
        model.fit(X_train, y_train)
    except Exception as e:
        if "OpenCL" in str(e) or "gpu" in str(e).lower():
            logger.warning("Final GPU fit failed (%s). Falling back to CPU.", e)
            best_params["device"] = "cpu"
            model = LGBMClassifier(**best_params)
            model.fit(X_train, y_train)
        else:
            raise

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    acc = accuracy_score(y_val, (preds > 0.5).astype(int))
    report = classification_report(y_val, (preds > 0.5).astype(int))
    logger.info("LightGBM Validation AUC: %.4f, Accuracy: %.4f\n%s", auc, acc, report)

    

    dump(model, output_model_path)
    logger.info("Saved LightGBM model → %s", output_model_path)

    importances = pd.Series(model.feature_importances_, index=features)
    dump(importances, output_importance_path)
    logger.info("Saved feature importances → %s", output_importance_path)

    # Log results
    metrics = {"AUC": auc, "Accuracy": acc, "Classification Report": report}
    log_training_results("LightGBM", metrics)

# --------------------------------------------------------------------------- #
# LSTM TRAINING – USE decision FROM CSV (NO RECOMPUTATION, NO LEAKAGE)
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)

def train_lstm(params: Dict[str, Any]):
    X_train, y_train, X_val, y_val, X_test, y_test, features, y_val_np, val_returns, scaler = load_and_preprocess_data(
        data_path=params["data_path"],
        target_column=params["target_column"],
        scale_features=params.get("scale_features", True),
        selected_features=params.get("selected_features"),
        test_size=params.get("test_size", 0.2),
        random_state=params.get("random_state", 42),
    )

    # Reshape for LSTM: (samples, timesteps, features)
    X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val_lstm = X_val.values.reshape((X_val.shape[0], 1, X_val.shape[1]))
    X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    def objective(trial):
        hidden_size = trial.suggest_int("hidden_size", 32, 256)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        model = AttentiveBiLSTM(
            input_size=X_train.shape[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            seq_len=params.get("time_steps", X_train_lstm.shape[1])
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_dataset = TensorDataset(torch.tensor(X_train_lstm, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(torch.tensor(X_val_lstm, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1))
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        for epoch in range(50):  # Fixed epochs for trial
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.float().view(-1))
                loss.backward()
                optimizer.step()

        model.eval()
        val_preds = []
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                val_preds.append(torch.sigmoid(outputs).cpu().numpy())

        if val_preds:
            val_preds = np.concatenate(val_preds).flatten()
            auc = roc_auc_score(y_val, val_preds)
        else:
            auc = 0.5
        return auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=params.get("n_trials", 50))

    best_params = study.best_params
    logger.info("Best LSTM params: %s (AUC: %.4f)", best_params, study.best_value)

    params.update({
        "hidden_size": best_params["hidden_size"],
        "num_layers": best_params["num_layers"],
        "dropout_rate": float(best_params["dropout"]),
        "learning_rate": best_params["learning_rate"],
        "batch_size": best_params["batch_size"],
    })

    # Retrain best model
    model = AttentiveBiLSTM(
        input_size=X_train.shape[-1],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout_rate"],
        seq_len=params["time_steps"]
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    train_dataset = TensorDataset(torch.tensor(X_train_lstm, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)

    for epoch in tqdm(range(params.get("epochs", 50))):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1))
            loss.backward()
            optimizer.step()

    # Validation evaluation
    model.eval()
    val_preds = []
    with torch.no_grad():
        val_dataset = TensorDataset(torch.tensor(X_val_lstm, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1))
        val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            val_preds.append(torch.sigmoid(outputs).cpu().numpy())

    val_preds = np.concatenate(val_preds).flatten()

    # Test evaluation
    model.eval()
    test_preds = []
    with torch.no_grad():
        test_dataset = TensorDataset(torch.tensor(X_test_lstm, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1))
        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"])
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            test_preds.append(torch.sigmoid(outputs).cpu().numpy())

    test_preds = np.concatenate(test_preds).flatten()
    test_acc = accuracy_score(y_test, (test_preds > 0.5).astype(int))
    logger.info("LSTM Test Accuracy: %.4f", test_acc)

    lstm_seq_len = (
        params.get("seq_len")
        or params.get("time_steps")
        or X_train_lstm.shape[1]
    )
    lstm_config = {
        "input_size": len(features),
        "hidden_size": params["hidden_size"],
        "num_layers": params["num_layers"],
        "dropout": float(params["dropout_rate"]),
        "seq_len": lstm_seq_len,
        "time_steps": lstm_seq_len,
        "feature_cols": list(features),
    }

    _persist_lstm_artifacts(model.state_dict(), lstm_config)

    # Log results
    auc = roc_auc_score(y_val, val_preds)
    acc = accuracy_score(y_val, (val_preds > 0.5).astype(int))
    report = classification_report(y_val, (val_preds > 0.5).astype(int))
    metrics = {"AUC": auc, "Accuracy": acc, "Classification Report": report}
    log_training_results("LSTM", metrics)

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X_train)
    joblib.dump(scaler, str(MODEL_DIR / "scaler.joblib"))
    joblib.dump(list(features), str(MODEL_DIR / "feature_order.joblib"))
    logger.info("Saved scaler.joblib and feature_order.joblib for LSTM")
# --------------------------------------------------------------------------- #
# PPO
# --------------------------------------------------------------------------- #

def train_ppo(params: Dict[str, Any]):
    data_file = params.get("data_path")
    model_path = params.get("model_path", "models/ppo.zip")
    output_path = params.get("output_model_path", "models/updated_ppo.zip")
    timesteps = params.get("ppo_timesteps", 20000)
    learning_rate = params.get("learning_rate", 0.0003)
    batch_size = params.get("batch_size", 64)
    n_steps = params.get("n_steps", 2048)
    clip_range = params.get("clip_range", 0.2)
    
    # Load data
    df = pd.read_csv(data_file, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    logger.info(f"Loaded {len(df)} rows from {data_file}")

    # Optional: Filter for specific ticker if needed (e.g., 'TSLA')
    # df = df[df['ticker'] == 'TSLA']
    # logger.info(f"Filtered for TSLA: {len(df)} rows")

    # Check row count
    if len(df) < MIN_SUPERVISED_ROWS:
        logger.warning(f"Only {len(df)} rows; skipping PPO training.")
        return

    # Split
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
    df_val, df_test = train_test_split(df_test, test_size=0.5, shuffle=False)
    logger.info(f"Split data: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    # Preprocess if needed - exclude price columns from scaling to preserve them for rewards
    if params.get("preprocessing", {}).get("scale_features", False):
        scaler = StandardScaler()
        excluded_cols = ["timestamp", "ticker", "price_1h", "price_4h", "price_1d"]  # Preserve prices
        features = [col for col in df_train.columns if col not in excluded_cols]
        df_train[features] = scaler.fit_transform(df_train[features])
        df_val[features] = scaler.transform(df_val[features])
        df_test[features] = scaler.transform(df_test[features])
        dump(scaler, MODEL_DIR / "ppo_scaler.joblib")
        logger.info("Features scaled and scaler saved")

    # Environments
    def make_env(df_subset):
        return lambda: TradingEnv(data=df_subset.copy())

    logger.info("Creating training environment...")
    train_env = DummyVecEnv([make_env(df_train)])
    logger.info("Training environment created")

    logger.info("Creating validation environment...")
    val_env = DummyVecEnv([make_env(df_val)])
    logger.info("Validation environment created")

    logger.info("Creating test environment...")
    test_env = DummyVecEnv([make_env(df_test)])
    logger.info("Test environment created")

    # Load or create model
    if os.path.exists(model_path):
        model = PPO.load(model_path, env=train_env,device="cpu")
        logger.info(f"Loaded existing PPO model from {model_path}")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_steps=n_steps,
            clip_range=clip_range,
            device='cpu'  # Force CPU to avoid GPU utilization issues with MlpPolicy
        )
        logger.info("Created new PPO model")

    # Train
    logger.info(f"Starting PPO training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)
    logger.info("PPO training completed")

    # Evaluate
    mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=10)
    logger.info(f"PPO Test Reward: {mean_reward:.4f} ± {std_reward:.4f}")

    # Save
    model.save(output_path)
    logger.info(f"Saved PPO model to {output_path}")

    # Log results
    metrics = {"mean_reward": mean_reward, "std_reward": std_reward}
    log_training_results("PPO", metrics)
# --------------------------------------------------------------------------- #
# transformer
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------
#  TRANSFORMER TRAINING – PYTORCH + OPTUNA + FP16
# --------------------------------------------------------------
logger = logging.getLogger(__name__)


def train_transformer(params: Dict[str, Any]):
    logger.info("=== STARTING FINAL GLOBAL TRANSFORMER TRAINING (PERFECT LABEL + NO LEAK) ===")

    # 1. LOAD YOUR GOLDEN DATA WITH THE PERFECT LABEL
    data_path = resolve_data_path("historical_data_no_price.csv")
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        return

    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    logger.info(f"Loaded {len(df):,} rows from {data_path.name}")

    # 2. FINAL SAFETY: REMOVE ANY RAW PRICE (should already be gone)
    price_cols = ["open", "high", "low", "close", "price", "adj_close"]
    dropped = [c for c in price_cols if c in df.columns]
    if dropped:
        logger.warning(f"Dropping raw price columns: {dropped}")
        df.drop(columns=dropped, inplace=True)

    # 3. USE YOUR PERFECT LABEL: decision (0=Buy, 1=Sell → from 8h forward return)
    if "decision" not in df.columns:
        logger.error("decision column missing! Run add_decision_column.py first!")
        return

    # Convert: 0=Buy → 1.0, 1=Sell → 0.0 (standard binary classification)
    y = (df["decision"] == 0).astype(np.float32).values  # 1.0 = Buy, 0.0 = Sell
    X = df[FEATURE_NAMES].fillna(0).values.astype(np.float32)

    seq_len = params["time_steps"]

    # 4. NO-LEAK SEQUENCE CREATION (safe for 8h forward label)
    def create_sequences_safe(X, y, seq_len):
        sequences = []
        labels = []
        for i in range(len(X) - seq_len - 7):  # -7 because we need 8 more bars after end
            sequences.append(X[i:i + seq_len])
            labels.append(y[i + seq_len - 1])  # label at the end of the sequence
        return np.array(sequences), np.array(labels)

    X_seq, y_seq = create_sequences_safe(X, y, seq_len)
    if len(X_seq) < 5000:
        logger.error("Not enough clean sequences! Check data.")
        return

    logger.info(f"Created {len(X_seq):,} clean sequences (8h forward label safe)")

    # 5. GLOBAL TRAIN/VAL SPLIT
    split_idx = int(len(X_seq) * 0.85)
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

    # 6. GLOBAL SCALER
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])

    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)

    # 7. DATASETS
    train_dataset = TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val_scaled), torch.from_numpy(y_val))

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)

    # 8. MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(
        input_size=len(FEATURE_NAMES),
        d_model=params["d_model"],
        nhead=params["nhead"],
        num_layers=params["num_layers"],
        dim_feedforward=params["dim_feedforward"],
        dropout=params["dropout"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.1))  # slight help for minority class
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=1e-5)

    best_val_auc = 0.0
    patience = 15
    patience_counter = 0

    for epoch in range(params["epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(yb.numpy())

        val_auc = roc_auc_score(all_labels, all_probs)
        logger.info(f"Epoch {epoch+1:3d} | Val AUC: {val_auc:.5f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_DIR / "updated_transformer.pt")
            joblib.dump(scaler, MODEL_DIR / "transformer_scaler.joblib")
            logger.info("  → NEW BEST MODEL SAVED")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping")
                break

    logger.info(f"Training complete! Best Val AUC: {best_val_auc:.5f}")
    logger.info("FINAL MODEL: updated_transformer.pt")
    logger.info("SCALER:      transformer_scaler.joblib")
    logger.info("=== TRANSFORMER IS NOW NUCLEAR ===")

# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def train_model(model_type: str, config_path: str):
    try:
        with open(config_path, "r") as f:
            params = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config {config_path}: {e}")
        return

    if not isinstance(params, dict):
        logger.error("Config must be a JSON object.")
        return

    if model_type == "RandomForest":
        train_random_forest(params)
    elif model_type == "XGBoost":
        train_xgboost(params)
    elif model_type == "LightGBM":
        train_lightgbm(params)
    elif model_type == "LSTM":
        train_lstm(params)
    elif model_type == "PPO":
        train_ppo(params)
    elif model_type == "Transformer":
        train_transformer(params)
    else:
        logger.error(f"Unsupported model_type: {model_type}")





def _save_pt_and_config(model, path_pt: Path, config: dict):
    torch.save(model.state_dict(), path_pt)
    config_path = path_pt.with_suffix(".json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved {path_pt.name} + {config_path.name}")

# --------------------------------------------------------------------------- #
# CLI entry-point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain a single model from its JSON config.")
    parser.add_argument("--config", required=True, help="Path to model config JSON")
    args = parser.parse_args()

   

    cfg_name = Path(args.config).name
    mapping = {
        "rf_config.json": "RandomForest",
        "xgb_config.json": "XGBoost",
        "lgb_config.json": "LightGBM",
        "lstm_config.json": "LSTM",
        "ppo_config.json": "PPO",
        "transformer_config.json": "Transformer"
    }
    mtype = mapping.get(cfg_name)
    if not mtype:
        logger.error(f"Unknown config file name: {cfg_name}")
    else:
        train_model(mtype, args.config)
    