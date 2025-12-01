# ml_predictor.py
# --------------------------------------------------------------
#  FULLY MATCHES retrain_models.py — LSTM + RF/XGB/LGB + PPO + Transformer
# --------------------------------------------------------------
import os
import logging
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import torch
from joblib import load
from stable_baselines3 import PPO

os.environ["SB3_PPO_WARN"] = "0"

# Import FEATURE_NAMES for fallback (matches retrain import) and surface
# candlestick/auxiliary feature constants for downstream utilities.
from self_learn import FEATURE_NAMES, CANDLESTICK_FEATURES, ADDITIONAL_FEATURES
from sequence_utils import pad_sequence_to_length

# ------------------------------------------------------------------
#  Project layout / shared artifacts
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Minimum confidence required for each model to trigger a trade (legacy thresholds
# kept for backward compatibility; the live ensemble below uses ENSEMBLE_THRESHOLDS).
MODEL_CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "RandomForest": 0.80,
    "XGBoost": 0.78,
    "LightGBM": 0.76,
    "PPO": 0.85,
    "LSTM": 0.985,
    "Transformer": 0.982,
}


# ==============================================================
# LIVE ENSEMBLE THRESHOLDS — THESE ARE THE REAL ONES (May–Nov 2025)
# ==============================================================
ENSEMBLE_THRESHOLDS: dict[str, float] = {
    "rf": 0.825,
    "xgb": 0.945,
    "lgb": 0.825,
    "lstm": 0.99,
    "transformer": 0.985,
}

# NUCLEAR CONFIRMATION — the entire reason we have 100% win rate
NUCLEAR_XGB  = 0.940
NUCLEAR_LSTM = 0.990


# Known aliases where legacy models saved during training expect the unsuffixed
# column, while the current live feature builder provides a duplicate with a
# ``.1`` suffix (or vice-versa). Keeping this mapping here lets every model
# loader harmonise the incoming dataframe without forcing upstream callers to
# remember to add the extra columns.
FEATURE_ALIASES: dict[str, str] = {
    "bb_position_1h.1": "bb_position_1h",
    "price_z_120h.1": "price_z_120h",
    "ret_1h.1": "ret_1h",
    "ret_4h.1": "ret_4h",
    "ret_24h.1": "ret_24h",
}

# Confidence threshold that forces a hold when high-confidence models disagree
HIGH_CONFIDENCE_DISAGREEMENT_THRESHOLD = 0.98

def _resolve_scaler_path() -> Optional[Path]:
    """Best-effort discovery of the StandardScaler used during training."""

    env_path = os.environ.get("SCALER_PATH")
    candidates = [
        Path(env_path) if env_path else None,
        MODEL_DIR / "scaler.joblib",
        MODEL_DIR / "scaler.pkl",
        Path("/app/models/scaler.pkl"),
        Path("/app/models/scaler.joblib"),
    ]

    for path in candidates:
        if path and path.exists():
            return path
    return None


_SCALER_CACHE = None
# Cache to avoid spamming identical warnings each time predict is called
_SCALER_MISMATCH_WARNED = False
_MISSING_FEATURE_LOG: dict[str, tuple[str, ...]] = {}
_ALIAS_WARNED = False
_TABULAR_FALLBACK_WARNED: set[str] = set()
_FEATURE_ORDER_CACHE = None


def _load_feature_scaler():
    """Return the cached StandardScaler (if available)."""

    global _SCALER_CACHE
    if _SCALER_CACHE is not None:
        return _SCALER_CACHE or None

    scaler_path = _resolve_scaler_path()
    if scaler_path is None:
        logging.warning("Feature scaler not found. Proceeding without scaling.")
        _SCALER_CACHE = False  # sentinel meaning "not available"
        return None

    try:
        _SCALER_CACHE = load(scaler_path)
        logging.info(f"Loaded feature scaler from {scaler_path}")
        return _SCALER_CACHE
    except Exception as exc:
        logging.error(f"Unable to load scaler from {scaler_path}: {exc}")
        _SCALER_CACHE = False
        return None


def _load_feature_order() -> Optional[List[str]]:
    """Return the feature order used during training if it was saved."""

    global _FEATURE_ORDER_CACHE
    if _FEATURE_ORDER_CACHE is not None:
        return _FEATURE_ORDER_CACHE or None

    env_path = os.environ.get("FEATURE_ORDER_PATH")
    candidates = [
        Path(env_path).expanduser() if env_path else None,
        MODEL_DIR / "feature_order.joblib",
        MODEL_DIR / "transformer_feature_order.joblib",
        Path("/app/models/feature_order.joblib"),
        Path("/app/models/transformer_feature_order.joblib"),
    ]

    for path in candidates:
        if path and path.exists():
            try:
                order = load(path)
                _FEATURE_ORDER_CACHE = [str(col) for col in order]
                logging.info(f"Loaded saved feature order from {path}")
                return _FEATURE_ORDER_CACHE
            except Exception as exc:
                logging.warning(f"Failed to load feature order from {path}: {exc}")

    _FEATURE_ORDER_CACHE = False
    return None


def _apply_feature_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Fill well-known alias columns so downstream models don't bail early."""

    global _ALIAS_WARNED
    missing_aliases = []
    updated = df.copy()

    for alias, source in FEATURE_ALIASES.items():
        if alias in updated.columns:
            continue
        if source in updated.columns:
            updated[alias] = updated[source]
            missing_aliases.append(alias)

    if missing_aliases and not _ALIAS_WARNED:
        _ALIAS_WARNED = True
        logging.info(
            "Filled missing alias columns to match legacy feature schema: %s",
            ", ".join(sorted(missing_aliases)),
        )

    return updated


def _prepare_feature_frame(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Ensure column order, clean inf/nan and apply the saved scaler once."""
    scaler = _load_feature_scaler()
    scaler_cols = getattr(scaler, "feature_names_in_", None) if scaler is not None else None

    # Build a canonical column order that keeps the scaler order first (if present)
    # and then appends any requested columns the scaler doesn't know about.
    canonical_columns: List[str]
    missing_for_scaler: List[str] = []
    new_columns: List[str] = []

    if scaler_cols is None:
        canonical_columns = list(dict.fromkeys(columns))
    else:
        scaler_cols = list(scaler_cols)
        requested_set = set(columns)
        scaler_set = set(scaler_cols)
        missing_for_scaler = [col for col in scaler_cols if col not in requested_set]
        new_columns = [col for col in columns if col not in scaler_set]

        canonical_columns = [
            *scaler_cols,
            *[col for col in columns if col not in scaler_set],
        ]

        if missing_for_scaler or new_columns:
            global _SCALER_MISMATCH_WARNED
            if not _SCALER_MISMATCH_WARNED:
                _SCALER_MISMATCH_WARNED = True
                logging.warning(
                    "Scaler feature set (%d columns) differs from requested frame (%d). Missing for scaler: %s | New columns: %s",
                    len(scaler_cols),
                    len(columns),
                    ", ".join(missing_for_scaler) or "None",
                    ", ".join(new_columns) or "None",
                )

    ordered = df.copy().reindex(columns=canonical_columns, fill_value=0)
    ordered = ordered.replace([np.inf, -np.inf], np.nan).fillna(0)
    ordered = ordered.infer_objects(copy=False)

    if scaler is None:
        return ordered.astype(float).reindex(columns=columns, fill_value=0)

    try:
        ordered_for_scaler = ordered.reindex(columns=scaler_cols, fill_value=0)
        scaled_values = scaler.transform(ordered_for_scaler.astype(float))
        scaled_df = pd.DataFrame(scaled_values, columns=scaler_cols, index=ordered.index)
        if new_columns:
            scaled_df = scaled_df.reindex(columns=canonical_columns, fill_value=0)
        return scaled_df.reindex(columns=columns, fill_value=0)
    except Exception as exc:
        logging.error(f"Scaler transform failed: {exc}")
        return ordered.astype(float).reindex(columns=columns, fill_value=0)

# Generic loader for joblib-persisted tabular models (RF/XGB/LGB).
def _load_joblib_model(path: Path, name: str):
    if not path.exists():
        return None, f"{name} missing"
    try:
        model = load(path)
        return model, None
    except Exception as exc:  # pragma: no cover - defensive logging
        return None, f"{name} load_error: {exc}"


def _predict_tabular_model(df: pd.DataFrame, name: str, path: Path) -> Tuple[np.ndarray, np.ndarray]:
    model, err = _load_joblib_model(path, name)
    if err:
        logging.warning(err)
        return np.array([]), np.array([])

    df = _apply_feature_aliases(_ensure_dataframe(df))

    try:
        cols: List[str]
        booster_names = None
        expected_n = getattr(model, "n_features_in_", None)

        # Prefer explicit training column names when available.
        feature_names_in = getattr(model, "feature_names_in_", None)
        if feature_names_in is not None:
            cols = list(feature_names_in)
        else:
            booster = getattr(model, "get_booster", None)
            if booster:
                try:
                    booster_names = booster().feature_names
                except Exception:
                    booster_names = None
            if booster_names:
                cols = list(booster_names)
            elif expected_n:
                cols = list(FEATURE_NAMES[:expected_n])
                if name not in _TABULAR_FALLBACK_WARNED:
                    _TABULAR_FALLBACK_WARNED.add(name)
                    logging.warning(
                        "%s missing feature_names_in_; using first %d FEATURE_NAMES to match training shape.",
                        name,
                        expected_n,
                    )
            else:
                cols = list(FEATURE_NAMES)

        if expected_n and len(cols) > expected_n:
            cols = cols[:expected_n]

        feature_df = _prepare_feature_frame(df, cols)
        row = feature_df.tail(1).astype(float).values

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(row)[:, 1]
        else:
            preds = model.predict(row)
            probs = np.asarray(preds, dtype=float)
            if probs.max() > 1 or probs.min() < 0:
                probs = 1 / (1 + np.exp(-probs))

        decisions = (probs > 0.5).astype(int)
        return probs, decisions
    except Exception as exc:
        logging.error(f"{name} predict error: {exc}")
        return np.array([]), np.array([])

# Canonical list of ensemble model names used across the trading stack.
MODEL_NAMES: tuple[str, ...] = (
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "PPO",
    "LSTM",
    "Transformer",
)
MODEL_DECISION_COLUMNS: tuple[str, ...] = tuple(f"{name}_decision" for name in MODEL_NAMES)

def _ensure_dataframe(data: object) -> pd.DataFrame:
    """Return a one-row dataframe regardless of the original input type."""

    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, pd.Series):
        return data.to_frame().T
    if isinstance(data, Mapping):
        return pd.DataFrame([data])
    try:
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame([data])


def predict_random_forest(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    df = _ensure_dataframe(df)
    return _predict_tabular_model(df, "RandomForest", MODEL_DIR / "rf_latest.joblib")


def predict_xgboost(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    df = _ensure_dataframe(df)
    return _predict_tabular_model(df, "XGBoost", MODEL_DIR / "xgb_latest.joblib")


def predict_lightgbm(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    df = _ensure_dataframe(df)
    return _predict_tabular_model(df, "LightGBM", MODEL_DIR / "lgb_latest.joblib")


def predict_lstm(df: pd.DataFrame, seq_len: int = 60):
    df = _apply_feature_aliases(_ensure_dataframe(df))
    model_path = MODEL_DIR / "lstm_best.pt"
    if not model_path.exists():
        logging.warning("LSTM model not found: %s", model_path)
        return np.array([]), np.array([])

    try:
        # 使用你当前真实的模型类
        from models.lstm import AttentiveBiLSTM
        model = AttentiveBiLSTM(input_size=len(FEATURE_NAMES))
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        logging.error(f"LSTM load failed: {e}")
        return np.array([]), np.array([])

    seq = pad_sequence_to_length(df, seq_len)
    if seq.shape[0] < seq_len:
        return np.array([]), np.array([])

    X = torch.tensor(seq.values, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, features)

    try:
        with torch.no_grad():
            logit = model(X).numpy()[0]
            prob = 1 / (1 + np.exp(-logit))  # sigmoid
            decision = 1 if prob > 0.5 else 0
            return np.array([prob]), np.array([decision])
    except Exception as e:
        logging.error(f"LSTM inference error: {e}")
        return np.array([]), np.array([])


def predict_transformer(df: pd.DataFrame, seq_len: int = 60):
    df = _apply_feature_aliases(_ensure_dataframe(df))
    model_path = MODEL_DIR / "transformer_best.pt"
    if not model_path.exists():
        logging.warning("Transformer model not found: %s", model_path)
        return np.array([]), np.array([])

    try:
        from models.transformer import TransformerModel
        model = TransformerModel(input_size=len(FEATURE_NAMES))
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        logging.error(f"Transformer load failed: {e}")
        return np.array([]), np.array([])

    seq = pad_sequence_to_length(df, seq_len)
    if seq.shape[0] < seq_len:
        return np.array([]), np.array([])

    X = torch.tensor(seq.values, dtype=torch.float32).unsqueeze(0)

    try:
        with torch.no_grad():
            logit = model(X).numpy()[0]
            prob = 1 / (1 + np.exp(-logit))
            decision = 1 if prob > 0.5 else 0
            return np.array([prob]), np.array([decision])
    except Exception as e:
        logging.error(f"Transformer inference error: {e}")
        return np.array([]), np.array([])


# ------------------------------------------------------------------
#  Tabular Models — EXACTLY MATCHES TRAINING
# ------------------------------------------------------------------
def _load_tabular(name: str):
    path = MODEL_DIR / f"updated_{name}.pkl"
    if not path.exists():
        return None, "missing"
    try:
        return load(path), None
    except Exception as e:
        return None, str(e)


def _resolve_model_features(model) -> List[str]:
    """Best-effort attempt to discover the feature order a model expects."""

    for attr in ("feature_names_in_", "feature_name_", "feature_names"):
        names = getattr(model, attr, None)
        if names is not None and len(names):
            return [str(col) for col in list(names)]

    booster = getattr(model, "get_booster", None)
    if callable(booster):
        try:
            booster_obj = booster()
            if booster_obj is not None and getattr(booster_obj, "feature_names", None):
                return list(booster_obj.feature_names)
        except Exception:
            pass

    return FEATURE_NAMES


def _log_missing_features(model_name: str, missing: List[str]):
    if not missing:
        return

    key = tuple(sorted(missing))
    if _MISSING_FEATURE_LOG.get(model_name) == key:
        return

    _MISSING_FEATURE_LOG[model_name] = key
    logging.warning(
        "%s model missing %d feature(s) that were zero-filled prior to prediction: %s",
        model_name,
        len(missing),
        ", ".join(missing)
    )


def predict_tabular(df: pd.DataFrame, name: str):
    model, err = _load_tabular(name)
    if err:
        logging.warning(f"{name} load failed: {err}")
        return np.array([]), np.array([])
    try:
        base_frame = _ensure_dataframe(df)
        feature_order = _resolve_model_features(model) or FEATURE_NAMES

        missing = [col for col in feature_order if col not in base_frame.columns]
        if missing:
            _log_missing_features(name, missing)

        X = _prepare_feature_frame(base_frame, feature_order)

        # Prefer probability outputs; fall back to class predictions if needed.
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X.values)[:, 1]
        else:
            preds = model.predict(X.values)
            probs = np.asarray(preds, dtype=float)
            # Guard against logits or labels by applying sigmoid when values
            # fall outside the probability range.
            if probs.max() > 1 or probs.min() < 0:
                probs = 1 / (1 + np.exp(-probs))

        decisions = (probs > 0.5).astype(int)
        return probs, decisions
    except Exception as e:
        logging.error(f"{name} predict error: {e}")
        return np.array([]), np.array([])


# ------------------------------------------------------------------
#  PPO — EXACTLY MATCHES TRAINING
# ------------------------------------------------------------------
def _load_ppo(path: Optional[Path] = None):
    path = path or (MODEL_DIR / "updated_ppo.zip")
    if not path.exists():
        return None, "missing"
    try:
        return PPO.load(path, device="cpu"), None
    except Exception as e:
        return None, str(e)


def predict_ppo(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    df = _ensure_dataframe(df)

    # Prefer the explicitly named trading agent checkpoint but fall back to legacy name.
    primary_path = MODEL_DIR / "ppo_trading_agent.zip"
    fallback_path = MODEL_DIR / "updated_ppo.zip"
    path = primary_path if primary_path.exists() else fallback_path

    model, err = _load_ppo(path)
    if err:
        logging.warning(f"PPO load failed: {err}")
        return np.array([]), np.array([])

    try:
        obs_frame = df.copy().reindex(columns=FEATURE_NAMES, fill_value=0)
        obs_frame = obs_frame.replace([np.inf, -np.inf], np.nan).fillna(0)
        obs_frame = obs_frame.infer_objects(copy=False)
        row = obs_frame.tail(1).values.astype(np.float32)
        obs_space = getattr(model, "observation_space", None)
        expected_len = int(obs_space.shape[0]) if getattr(obs_space, "shape", None) else row.shape[1]
        if row.shape[1] < expected_len:
            pad = expected_len - row.shape[1]
            row = np.pad(row, ((0, 0), (0, pad)), constant_values=0.0)
        elif row.shape[1] > expected_len:
            row = row[:, :expected_len]

        action, _ = model.predict(row, deterministic=True)
        prob = 0.8 if action[0] == 1 else 0.2
        probs = np.array([prob])
        decisions = action.flatten().astype(int)
        return probs, decisions
    except Exception as e:
        logging.error(f"PPO load/predict failed: {e}")
        return np.array([]), np.array([])


# ------------------------------------------------------------------
#  PUBLIC API — MATCHES tsla_ai_master_final_ready_backtest.py
# ------------------------------------------------------------------
def predict_with_all_models(df: pd.DataFrame, seq_len: int = 60) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    df = _ensure_dataframe(df)

    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    probs, dec = predict_random_forest(df)
    if len(probs):
        results["RandomForest"] = (probs, dec)

    probs, dec = predict_xgboost(df)
    if len(probs):
        results["XGBoost"] = (probs, dec)

    probs, dec = predict_lightgbm(df)
    if len(probs):
        results["LightGBM"] = (probs, dec)

    probs, dec = predict_ppo(df)
    if len(probs):
        results["PPO"] = (probs, dec)

    # Sequence models
    probs, dec = predict_lstm(df, seq_len=seq_len)
    if len(probs):
        results["LSTM"] = (probs, dec)

    probs, dec = predict_transformer(df, seq_len=seq_len)
    if len(probs):
        results["Transformer"] = (probs, dec)

    return results


def _extract_vote_and_confidence(probs: np.ndarray) -> Tuple[Optional[str], float]:
    """Return vote (Buy/Sell/Hold) and confidence from probability array."""

    if probs.size == 0:
        return None, 0.0

    finite = np.asarray(probs, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None, 0.0

    non_zero = finite[finite != 0]
    effective = non_zero if non_zero.size else finite

    prob = float(effective.mean())
    if prob > 0.5:
        return "Buy", prob
    if prob < 0.5:
        return "Sell", 1 - prob
    return "Hold", 0.5


def _get_row_value(row: Mapping, key: str, default=None):
    if hasattr(row, "get"):
        return row.get(key, default)
    try:
        return row[key]
    except Exception:
        return default


def get_ensemble_signal(row):
    votes = []
    confs = {}

    if (_get_row_value(row, "rf_conf", 0.0) >= ENSEMBLE_THRESHOLDS["rf"]):
        vote = _get_row_value(row, "rf_vote")
        if vote:
            votes.append(vote)
            confs["rf"] = _get_row_value(row, "rf_conf", 0.0)

    if (_get_row_value(row, "xgb_conf", 0.0) >= ENSEMBLE_THRESHOLDS["xgb"]):
        vote = _get_row_value(row, "xgb_vote")
        if vote:
            votes.append(vote)
            confs["xgb"] = _get_row_value(row, "xgb_conf", 0.0)

    if (_get_row_value(row, "lgb_conf", 0.0) >= ENSEMBLE_THRESHOLDS["lgb"]):
        vote = _get_row_value(row, "lgb_vote")
        if vote:
            votes.append(vote)
            confs["lgb"] = _get_row_value(row, "lgb_conf", 0.0)

    if (_get_row_value(row, "lstm_conf", 0.0) >= ENSEMBLE_THRESHOLDS["lstm"]):
        vote = _get_row_value(row, "lstm_vote")
        if vote:
            votes.append(vote)
            confs["lstm"] = _get_row_value(row, "lstm_conf", 0.0)

    if (_get_row_value(row, "transformer_conf", 0.0) >= ENSEMBLE_THRESHOLDS["transformer"]):
        vote = _get_row_value(row, "transformer_vote")
        if vote:
            votes.append(vote)
            confs["transformer"] = _get_row_value(row, "transformer_conf", 0.0)

    if (_get_row_value(row, "ppo_conf", 0.0) >= ENSEMBLE_THRESHOLDS["ppo"]):
        vote = _get_row_value(row, "ppo_vote")
        if vote:
            votes.append(vote)
            confs["ppo"] = _get_row_value(row, "ppo_conf", 0.0)

    if len(votes) < 3:
        return "HOLD", 0.0, votes, confs

    buy_count = votes.count("Buy")
    sell_count = votes.count("Sell")
    majority = "Buy" if buy_count > sell_count else "Sell" if sell_count > buy_count else "HOLD"

    if majority == "HOLD":
        return "HOLD", 0.0, votes, confs

    nuclear_ok = False
    if (
        ("xgb" in confs and confs["xgb"] >= 0.94 and _get_row_value(row, "xgb_vote") == majority)
        or ("lstm" in confs and confs["lstm"] >= 0.99 and _get_row_value(row, "lstm_vote") == majority)
    ):
        nuclear_ok = True

    if not nuclear_ok:
        return "HOLD", 0.0, votes, confs

    avg_conf = np.mean([confs[k] for k, v in confs.items() if _get_row_value(row, f"{k}_vote") == majority])
    return majority, float(avg_conf), votes, confs


def get_position_size_and_actions(row, prev_row, current_signal, current_position):
    if pd.isna(_get_row_value(row, "ppo_action", np.nan)) or current_signal == "HOLD":
        return 1.0, False, False

    action = int(_get_row_value(row, "ppo_action", 1))
    value = _get_row_value(row, "ppo_value", 0)
    entropy = _get_row_value(row, "ppo_entropy", 0.3)

    value_ma = _get_row_value(row, "ppo_value_ma100", value)
    value_std = _get_row_value(row, "ppo_value_std100", 0.5)

    allow_pyramid = False
    early_exit = False
    target_size = 1.0

    if action == 2 and value > value_ma + 0.5 * value_std and entropy < 0.4:
        allow_pyramid = True
        target_size = 1.75

    if action == 0 or entropy > 0.60:
        early_exit = True

    if current_position != 0 and action == 0 and _get_row_value(prev_row, "ppo_action", 1) == 2:
        early_exit = True

    return target_size, allow_pyramid, early_exit


def _build_live_row(predictions: Dict[str, Tuple[np.ndarray, np.ndarray]], ppo_metadata: Optional[Mapping]) -> Tuple[dict, dict]:
    detail = {
        "votes": {},
        "confidences": {},
        "qualified": [],
        "missing": {},
    }
    row: dict = {}

    model_map = {
        "RandomForest": "rf",
        "XGBoost": "xgb",
        "LightGBM": "lgb",
        "LSTM": "lstm",
        "Transformer": "transformer",
        "PPO": "ppo",
    }

    for model_name, short in model_map.items():
        if model_name not in predictions:
            detail["missing"][model_name] = "no prediction"
            continue

        probs, _ = predictions[model_name]
        vote, conf = _extract_vote_and_confidence(probs)
        if vote is None:
            detail["missing"][model_name] = "invalid probabilities"
            continue

        detail["votes"][model_name] = vote
        detail["confidences"][model_name] = conf
        row[f"{short}_vote"] = vote
        row[f"{short}_conf"] = conf
        if conf >= ENSEMBLE_THRESHOLDS.get(short, 1.0) and vote != "Hold":
            detail["qualified"].append(model_name)

    ppo_action = None
    if "PPO" in predictions:
        _, decisions = predictions["PPO"]
        if decisions.size:
            ppo_action = int(decisions.flatten()[-1])

    if ppo_metadata:
        row["ppo_value"] = ppo_metadata.get("value")
        row["ppo_entropy"] = ppo_metadata.get("entropy")
        row["ppo_value_ma100"] = ppo_metadata.get("value_ma100")
        row["ppo_value_std100"] = ppo_metadata.get("value_std100")
        ppo_action = ppo_metadata.get("action", ppo_action)

    if ppo_action is not None:
        row["ppo_action"] = ppo_action

    return row, detail


def _live_ensemble_decision(
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    return_details: bool,
    current_position: float,
    prev_row: Optional[Mapping],
    ppo_metadata: Optional[Mapping],
):
    row, detail = _build_live_row(predictions, ppo_metadata)
    signal, ensemble_conf, votes, confs = get_ensemble_signal(row)

    prev_row = prev_row or {}
    target_size, allow_pyramid, early_exit = get_position_size_and_actions(
        row, prev_row, signal, current_position
    )

    direction = 0
    if signal == "Buy":
        direction = 1
    elif signal == "Sell":
        direction = -1

    position_size = 0.0 if (signal == "HOLD" or early_exit) else direction * target_size

    detail.update(
        {
            "ensemble_signal": signal,
            "ensemble_confidence": ensemble_conf,
            "qualified_models": list(confs.keys()),
            "votes_sequence": votes,
            "pyramid_allowed": allow_pyramid,
            "early_exit": early_exit,
            "position_size": position_size,
            "target_size": target_size,
            "thresholds": ENSEMBLE_THRESHOLDS,
        }
    )

    if len(votes) < 3:
        detail["reason"] = "Less than 3 qualified models — HOLD"
    elif signal == "HOLD" and votes:
        detail["reason"] = "Majority tie — HOLD"
    elif signal == "HOLD":
        detail["reason"] = "Nuclear confirmation failed — HOLD"
    else:
        detail["reason"] = (
            f"Majority {signal} with nuclear confirmation; avg_conf={ensemble_conf:.3f}"
        )

    final = signal if signal != "HOLD" else "Hold"
    if return_details:
        return final, detail
    return final


NUCLEAR_XGB = 0.94
NUCLEAR_LSTM = 0.99

def independent_model_decisions(predictions: Dict, return_details: bool = False):
    detail = {
        "votes": {},
        "confidences": {},
        "thresholds": ENSEMBLE_THRESHOLDS.copy(),
        "triggers": [],
        "missing": {},
    }

    # Step 1: Collect only models that beat their personal threshold
    qualified = []
    for name, (probs, extra) in predictions.items():
        if name not in ENSEMBLE_THRESHOLDS:
            continue
        thresh = ENSEMBLE_THRESHOLDS[name]

        if probs is None or len(probs) == 0:
            detail["missing"][name] = "no prediction"
            continue

        prob = float(np.mean([p for p in probs if np.isfinite(p) and p != 0]) or 0.5)
        decision = "Buy" if prob > 0.5 else "Sell" if prob < 0.5 else "Hold"
        confidence = prob if decision == "Buy" else 1 - prob

        detail["votes"][name] = decision
        detail["confidences"][name] = confidence

        if confidence >= thresh and decision != "Hold":
            qualified.append((name, decision, confidence))

    if len(qualified) < 3:
        final = "Hold"
        detail["reason"] = f"Only {len(qualified)} models qualified (<3)"
        return (final, detail) if return_details else final

    # Step 2: Majority vote
    votes = [dec for _, dec, _ in qualified]
    majority = "Buy" if votes.count("Buy") > votes.count("Sell") else "Sell"

    # Step 3: NUCLEAR CONFIRMATION — at least one XGBoost ≥0.94 or LSTM ≥0.99 must agree
    nuclear_ok = False
    for name, dec, conf in qualified:
        if name == "xgb" and conf >= NUCLEAR_XGB and dec == majority:
            nuclear_ok = True
        if name == "lstm" and conf >= NUCLEAR_LSTM and dec == majority:
            nuclear_ok = True
    if not nuclear_ok:
        final = "Hold"
        detail["reason"] = "No nuclear confirmation (XGB≥0.94 or LSTM≥0.99)"
        return (final, detail) if return_details else final

    avg_conf = np.mean([conf for _, dec, conf in qualified if dec == majority])
    detail["reason"] = f"≥3 qualified + nuclear confirmed → {majority}"
    detail["confidence"] = float(avg_conf)

    # Attach PPO data for live script (pyramiding & early exit)
    if "ppo" in predictions:
        ppo_probs, ppo_extra = predictions["ppo"]
        detail["ppo_action"] = int(ppo_extra.get("action", 1))      # 0=reduce, 1=hold, 2=aggressive
        detail["ppo_value"] = float(ppo_extra.get("value", 0.0))
        detail["ppo_entropy"] = float(ppo_extra.get("entropy", 0.3))
        detail["ppo_value_ma100"] = float(ppo_extra.get("value_ma100", 0.0))
        detail["ppo_value_std100"] = float(ppo_extra.get("value_std100", 0.5))

    return (majority, detail) if return_details else majority


# ------------------------------------------------------------------
#  CLI (optional)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to input CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    preds = predict_with_all_models(df)
    decision, info = independent_model_decisions(preds, return_details=True)
    print(f"FINAL: {decision}")
    print(json.dumps(info, indent=2))
