# ml_predictor.py
# --------------------------------------------------------------
#  FULLY MATCHES retrain_models.py — LSTM + RF/XGB/LGB + PPO + Transformer
# --------------------------------------------------------------
import os
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

# Confidence threshold that forces a hold when high-confidence models disagree
HIGH_CONFIDENCE_DISAGREEMENT_THRESHOLD = 0.98

# Live-trading ensemble thresholds (battle-tested May–Nov 2025)
ENSEMBLE_THRESHOLDS: dict[str, float] = {
    "rf": 0.825,
    "xgb": 0.945,
    "lgb": 0.825,
    "lstm": 0.99,
    "transformer": 0.985,
    "ppo": 0.80,
}


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

    try:
        cols = list(getattr(model, "feature_names_in_", FEATURE_NAMES))
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

# ------------------------------------------------------------------
#  LSTM Model — EXACTLY MATCHES TRAINING
# ------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


def _infer_lstm_cfg_from_state(pt_path: Path) -> Optional[Dict]:
    """Best-effort inference of LSTM hyperparameters from the saved weights."""

    try:
        state_dict = torch.load(pt_path, map_location="cpu")
    except Exception as exc:
        logging.error(f"Unable to inspect LSTM weights: {exc}")
        return None

    weight_keys = [k for k in state_dict.keys() if k.startswith("lstm.weight_ih_l")]
    if not weight_keys:
        logging.error("LSTM state dict missing expected weight_ih entries")
        return None

    first_weight = state_dict[weight_keys[0]]
    if first_weight.ndim != 2:
        logging.error("Unexpected LSTM weight tensor shape: %s", tuple(first_weight.shape))
        return None

    hidden_size = first_weight.shape[0] // 4
    input_size = first_weight.shape[1]

    layer_indices: List[int] = []
    for key in weight_keys:
        suffix = key.split("lstm.weight_ih_l", 1)[1]
        layer_token = suffix.split("_")[0]
        if layer_token.isdigit():
            layer_indices.append(int(layer_token))

    num_layers = (max(layer_indices) + 1) if layer_indices else 1

    cfg = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": 0.0,
        "seq_len": 1,
        "time_steps": 1,
        "feature_cols": FEATURE_NAMES,
    }
    return cfg


def _load_lstm() -> Tuple[Optional[nn.Module], Optional[Dict], Optional[str]]:
    pt_path = MODEL_DIR / "updated_lstm.pt"
    cfg_path = MODEL_DIR / "updated_lstm.json"

    if not pt_path.exists():
        return None, None, "missing_pt"

    # First, try to load a fully-serialized model (structure + weights)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        full_model = torch.load(pt_path, map_location=device)
        if isinstance(full_model, nn.Module):
            full_model.eval()
            logging.info(f"LSTM loaded as full model from {pt_path}")
            return full_model, None, None
    except Exception as exc:
        logging.debug(
            "Failed to load LSTM checkpoint as a full model (will try state_dict): %s",
            exc,
        )

    cfg = None
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except Exception as e:
            return None, None, f"cfg_error: {e}"
    else:
        logging.warning("LSTM config missing; attempting to infer from weights.")
        cfg = _infer_lstm_cfg_from_state(pt_path)
        if cfg is None:
            return None, None, "missing_cfg"
        try:
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            logging.info(f"Inferred LSTM config saved → {cfg_path}")
        except Exception as exc:
            logging.warning(f"Failed to persist inferred LSTM config: {exc}")

    try:
        model = LSTMModel(
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg.get("dropout", 0.0)
        )
        state_dict = torch.load(pt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logging.info(f"LSTM loaded via state_dict from {pt_path}")
        return model, cfg, None
    except Exception as e:
        return None, None, f"load_error: {e}"


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


def predict_lstm(df: pd.DataFrame, seq_len: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    df = _ensure_dataframe(df)

    model, cfg, err = _load_lstm()
    if err:
        logging.warning(f"LSTM load failed: {err}")
        return np.array([]), np.array([])

    cols = cfg.get("feature_cols", FEATURE_NAMES)  # Matches retrain save (has feature_cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logging.error(f"LSTM missing features: {missing[:5]}...")
        return np.array([]), np.array([])

    # Build sequence for LSTM — ALWAYS use latest CLOSED bar as the final (most recent) entry
    use_seq_len = seq_len or cfg.get("seq_len") or cfg.get("time_steps") or 60
    try:
        use_seq_len = int(use_seq_len)
    except (TypeError, ValueError):
        use_seq_len = 60

    sorted_df = df.sort_index()
    feature_df = _prepare_feature_frame(sorted_df, cols)
    df_seq = pad_sequence_to_length(
        feature_df,
        target_length=use_seq_len,
        feature_columns=cols,
    )

    expected_features = len(cols)
    actual_features = df_seq.shape[1] if df_seq.ndim == 2 else df_seq.shape[-1]
    if actual_features != expected_features:
        logging.error(
            "Feature count mismatch in LSTM: expected %d, got %d. Check live feature builder vs training.",
            expected_features,
            actual_features,
        )
        return np.array([]), np.array([])

    X = df_seq.reshape((1, use_seq_len, -1))  # (1 batch, seq_len, features) for single prediction

    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        tensor = torch.tensor(X, dtype=torch.float32, device=device)
        outputs = model(tensor)

        # Ensure probabilities even if the saved model emits logits.
        if torch.max(outputs) > 1 or torch.min(outputs) < 0:
            outputs = torch.sigmoid(outputs)

        probs = outputs.cpu().numpy().flatten()
        decisions = (probs > 0.5).astype(int)
    return probs, decisions


# ------------------------------------------------------------------
#  Transformer Model — EXACTLY MATCHES TRAINING
# ------------------------------------------------------------------
class TransformerModel(nn.Module):
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # Last token
        x = self.fc(x)
        return self.sigmoid(x)


def _load_transformer() -> Tuple[Optional[nn.Module], Optional[Dict], Optional[str]]:
    pt_path = MODEL_DIR / "updated_transformer.pt"
    cfg_path = MODEL_DIR / "updated_transformer.json"
    if not pt_path.exists():
        return None, None, "missing_pt"
    if not cfg_path.exists():
        return None, None, "missing_cfg"

    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception as e:
        return None, None, f"cfg_error: {e}"

    try:
        model = TransformerModel(
            input_size=cfg["input_size"],
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg.get("dropout", 0.0)
        )
        state_dict = torch.load(pt_path, map_location="cpu")

        # Remap legacy layer names (e.g., "embed" -> "embedding", "classifier" -> "fc")
        remapped_state = {}
        for old_key, value in state_dict.items():
            new_key = old_key.replace("embed", "embedding").replace("classifier", "fc")
            remapped_state[new_key] = value

        model.load_state_dict(remapped_state)
        model.eval()
        return model, cfg, None
    except Exception as e:
        return None, None, f"load_error: {e}"


def predict_transformer(df: pd.DataFrame, seq_len: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    df = _ensure_dataframe(df)

    model, cfg, err = _load_transformer()
    if err:
        logging.warning(f"Transformer load failed: {err}")
        return np.array([]), np.array([])

    cols = cfg.get("feature_cols", FEATURE_NAMES)  # Fallback if not saved (retrain doesn't save it)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logging.error(f"Transformer missing features: {missing[:5]}...")
        return np.array([]), np.array([])

    # Use seq_len from cfg or param (matches retrain's time_steps)
    use_seq_len = seq_len or cfg.get("time_steps", 1)
    sorted_df = df.sort_index()
    feature_df = _prepare_feature_frame(sorted_df, cols)
    df_seq = pad_sequence_to_length(
        feature_df,
        target_length=use_seq_len,
        feature_columns=cols,
    )

    expected_features = len(cols)
    actual_features = df_seq.shape[1] if df_seq.ndim == 2 else df_seq.shape[-1]
    if actual_features != expected_features:
        logging.error(
            "Feature count mismatch in Transformer: expected %d, got %d. Check live feature builder vs training.",
            expected_features,
            actual_features,
        )
        return np.array([]), np.array([])

    X = df_seq.reshape((1, use_seq_len, -1))

    with torch.no_grad():
        outputs = model(torch.tensor(X))

        # Ensure probabilities even if the saved model emits logits.
        if torch.max(outputs) > 1 or torch.min(outputs) < 0:
            outputs = torch.sigmoid(outputs)

        probs = outputs.cpu().numpy().flatten()
        decisions = (probs > 0.5).astype(int)
    return probs, decisions


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


def independent_model_decisions(predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
                               return_details: bool = False,
                               confidence_threshold: float = 0.98,
                               model_thresholds: Optional[Mapping[str, float]] = None,
                               current_position: float = 0.0,
                               prev_row: Optional[Mapping] = None,
                               ppo_metadata: Optional[Mapping] = None,
                               use_live_ensemble: bool = True) -> Tuple[str, Dict]:
    """Return trade decisions using the live ensemble (default) or legacy rules."""

    if use_live_ensemble:
        return _live_ensemble_decision(
            predictions,
            return_details,
            current_position,
            prev_row,
            ppo_metadata,
        )

    # Legacy fallback (kept for compatibility with earlier backtests). Any model
    # whose confidence meets or exceeds its own threshold will trigger its trade
    # decision. The function returns the first triggered decision (in a fixed
    # priority order) for backward compatibility, along with a details payload
    # describing all triggered models.

    detail = {
        "votes": {},
        "confidences": {},
        "missing": {},
        "thresholds": {},
        "trigger": None,
        "triggers": [],
        "reason": "",
        "confidence": 0.0,
    }

    ordered_models = list(MODEL_NAMES)

    thresholds_lookup = MODEL_CONFIDENCE_THRESHOLDS.copy()
    if model_thresholds:
        for key, value in model_thresholds.items():
            if value is None:
                continue
            for candidate in ordered_models:
                if candidate.lower() == key.lower():
                    thresholds_lookup[candidate] = float(value)
                    break

    def _threshold_for(model_name: str) -> float:
        return thresholds_lookup.get(model_name, confidence_threshold)

    for name in ordered_models:
        if name not in predictions:
            detail["missing"][name] = "no prediction"
            continue

        probs, _ = predictions[name]
        if probs.size == 0:
            detail["missing"][name] = "no prediction"
            continue

        finite = np.asarray(probs, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            detail["missing"][name] = "invalid probabilities"
            continue

        non_zero = finite[finite != 0]
        effective = non_zero if non_zero.size else finite

        prob = float(effective.mean())
        if prob > 0.5:
            decision = "Buy"
            confidence = prob
        elif prob < 0.5:
            decision = "Sell"
            confidence = 1 - prob
        else:
            decision = "Hold"
            confidence = 0.5

        detail["votes"][name] = decision
        detail["confidences"][name] = confidence
        threshold = _threshold_for(name)
        detail["thresholds"][name] = threshold

        if confidence >= threshold and decision != "Hold":
            trigger_info = {
                "model": name,
                "decision": decision,
                "confidence": confidence,
                "threshold": threshold,
            }
            detail["triggers"].append(trigger_info)

    # If both flagship models are highly confident but disagree, stand down
    if (
        set(detail["votes"].keys()) >= {"LSTM", "Transformer"}
        and detail["votes"]["LSTM"] != detail["votes"]["Transformer"]
        and detail["votes"]["LSTM"] != "Hold"
        and detail["votes"]["Transformer"] != "Hold"
        and detail["confidences"].get("LSTM", 0.0)
        >= HIGH_CONFIDENCE_DISAGREEMENT_THRESHOLD
        and detail["confidences"].get("Transformer", 0.0)
        >= HIGH_CONFIDENCE_DISAGREEMENT_THRESHOLD
    ):
        final = "Hold"
        detail["trigger"] = "LSTM vs Transformer"
        detail["confidence"] = min(
            detail["confidences"].get("LSTM", 0.0),
            detail["confidences"].get("Transformer", 0.0),
        )
        detail["reason"] = "HIGH CONFIDENCE DISAGREEMENT — HOLD"
        return (final, detail) if return_details else final

    if not detail["votes"]:
        final = "Hold"
        detail["reason"] = "No valid predictions available"
        return (final, detail) if return_details else final

    if detail["triggers"]:
        first = detail["triggers"][0]
        final = first["decision"]
        detail["trigger"] = first["model"]
        detail["confidence"] = first["confidence"]
        detail["reason"] = (
            f"{first['model']} confidence {first['confidence']:.2f} >= "
            f"{first['threshold']:.2f}; executing its decision"
        )
        return (final, detail) if return_details else final

    # If LSTM and Transformer agree on a non-hold direction, execute that
    # decision even if neither model individually cleared its own threshold.
    if (
        "LSTM" in detail["votes"]
        and "Transformer" in detail["votes"]
        and detail["votes"]["LSTM"] == detail["votes"]["Transformer"]
        and detail["votes"]["LSTM"] != "Hold"
    ):
        final = detail["votes"]["LSTM"]
        detail["trigger"] = "LSTM+Transformer"
        detail["confidence"] = min(
            detail["confidences"].get("LSTM", 0.0),
            detail["confidences"].get("Transformer", 0.0),
        )
        detail["reason"] = (
            "LSTM and Transformer agree on direction; executing consensus trade"
        )
        return (final, detail) if return_details else final

    final = "Hold"
    leader, leader_conf = max(detail["confidences"].items(), key=lambda item: item[1])
    detail["trigger"] = leader
    detail["confidence"] = leader_conf
    detail["reason"] = (
        f"No model met its confidence threshold; highest was {leader} "
        f"({leader_conf:.2f}/{detail['thresholds'].get(leader, confidence_threshold):.2f})"
    )
    return (final, detail) if return_details else final


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
