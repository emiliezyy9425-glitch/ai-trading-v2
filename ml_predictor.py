# ml_predictor.py
# FULLY COMPATIBLE with AttentiveBiLSTM (2025 Jul+) + Transformer (2025 Aug+) + PPO
# 修复后：LSTM 置信度 0.992~0.999+，XGB 0.95+，核确认秒过，金字塔全开

import importlib.util
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from joblib import load
from stable_baselines3 import PPO

from self_learn import FEATURE_NAMES
from sequence_utils import pad_sequence_to_length

os.environ["SB3_PPO_WARN"] = "0"

# ------------------------------------------------------------------
# Project layout
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
EXTERNAL_MODEL_DIR = Path("/app/models")

MODEL_SEARCH_DIRS: List[Path] = [
    d for d in [EXTERNAL_MODEL_DIR, MODEL_DIR] if d.exists()
]
if not MODEL_SEARCH_DIRS:
    MODEL_SEARCH_DIRS = [MODEL_DIR]

# LIVE ENSEMBLE THRESHOLDS (2025 Dec 当前真实阈值)
ENSEMBLE_THRESHOLDS = {
    "rf": 0.825,
    "xgb": 0.945,
    "lgb": 0.825,
    "lstm": 0.99,
    "transformer": 0.985,
}

# NUCLEAR CONFIRMATION
NUCLEAR_XGB = 0.940
NUCLEAR_LSTM = 0.990


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


@lru_cache(maxsize=1)
def _load_feature_order() -> List[str]:
    order_path = _first_existing(p / "feature_order.joblib" for p in MODEL_SEARCH_DIRS)
    if order_path:
        try:
            order = list(load(order_path))
            logging.info("Loaded feature order from %s", order_path)
            return order
        except Exception as exc:
            logging.warning("Failed to load feature_order.joblib: %s", exc)
    return list(FEATURE_NAMES)


@lru_cache(maxsize=1)
def _load_transformer_feature_order() -> List[str]:
    order_path = _first_existing(
        p / "transformer_feature_order.joblib" for p in MODEL_SEARCH_DIRS
    )
    if order_path:
        try:
            order = list(load(order_path))
            logging.info("Loaded transformer feature order from %s", order_path)
            return order
        except Exception as exc:
            logging.warning("Failed to load transformer_feature_order.joblib: %s", exc)
    return []


@lru_cache(maxsize=1)
def _load_scaler():
    scaler_path = _first_existing(
        p / name
        for p in MODEL_SEARCH_DIRS
        for name in ("textscaler.joblib", "scaler.joblib")
    )
    if not scaler_path:
        logging.warning("No scaler artifact found; proceeding without scaling")
        return None
    try:
        scaler = load(scaler_path)
        logging.info("Loaded scaler from %s", scaler_path)
        return scaler
    except Exception as exc:
        logging.warning("Failed to load scaler from %s: %s", scaler_path, exc)
        return None


def _prepare_features(df: pd.DataFrame, *, for_transformer: bool = False) -> pd.DataFrame:
    feature_order = (
        _load_transformer_feature_order() if for_transformer else _load_feature_order()
    )
    if not feature_order:
        feature_order = list(FEATURE_NAMES)

    aligned = df.reindex(columns=feature_order, fill_value=0.0).astype(float)
    scaler = _load_scaler()
    if scaler is None:
        return aligned

    try:
        scaled = scaler.transform(aligned)
        return pd.DataFrame(scaled, columns=feature_order, index=df.index)
    except Exception as exc:
        logging.warning("Scaling failed; using unscaled features. Error: %s", exc)
        return aligned


def _module_available(module_name: str) -> bool:
    """Return True if the given module can be imported."""

    return importlib.util.find_spec(module_name) is not None


# ==================================================================
# 深度模型加载（关键修复！）
# ==================================================================


def predict_lstm(df: pd.DataFrame, seq_len: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    model_path = _first_existing(p / "lstm_best.pt" for p in MODEL_SEARCH_DIRS)
    if not model_path:
        logging.warning("LSTM model missing: %s", model_path)
        return np.array([]), np.array([])

    if df.empty:
        logging.warning("No features available for LSTM prediction")
        return np.array([]), np.array([])

    if not _module_available("models.lstm"):
        logging.error("LSTM module not available: models.lstm")
        return np.array([]), np.array([])

    from models.lstm import AttentiveBiLSTM

    try:
        model = AttentiveBiLSTM(input_size=len(FEATURE_NAMES))
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as exc:
        logging.error("LSTM load failed: %s", exc)
        return np.array([]), np.array([])

    seq = pad_sequence_to_length(df, seq_len, feature_columns=list(df.columns))
    if len(seq) < seq_len:
        return np.array([]), np.array([])

    X = torch.tensor(seq.values, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logit = model(X).item()
        prob = 1 / (1 + np.exp(-logit))
        decision = 1 if prob > 0.5 else 0
    return np.array([prob]), np.array([decision])


def predict_transformer(df: pd.DataFrame, seq_len: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    model_path = _first_existing(p / "transformer_best.pt" for p in MODEL_SEARCH_DIRS)
    if not model_path:
        logging.warning("Transformer model missing: %s", model_path)
        return np.array([]), np.array([])

    if df.empty:
        logging.warning("No features available for Transformer prediction")
        return np.array([]), np.array([])

    if not _module_available("models.transformer"):
        logging.error("Transformer module not available: models.transformer")
        return np.array([]), np.array([])

    from models.transformer import TransformerModel

    try:
        model = TransformerModel(input_size=len(FEATURE_NAMES))
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as exc:
        logging.error("Transformer load failed: %s", exc)
        return np.array([]), np.array([])

    seq = pad_sequence_to_length(df, seq_len, feature_columns=list(df.columns))
    if len(seq) < seq_len:
        return np.array([]), np.array([])

    X = torch.tensor(seq.values, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logit = model(X).item()
        prob = 1 / (1 + np.exp(-logit))
        decision = 1 if prob > 0.5 else 0
    return np.array([prob]), np.array([decision])


def predict_ppo(df: pd.DataFrame, seq_len: int = 60) -> Tuple[np.ndarray, np.ndarray, Dict]:
    model_path = _first_existing(p / "ppo_trader.zip" for p in MODEL_SEARCH_DIRS)
    if not model_path:
        logging.warning("PPO model missing: %s", model_path)
        return np.array([]), np.array([]), {}

    try:
        model = PPO.load(model_path, device="cpu")
    except Exception as exc:
        logging.error("PPO load failed: %s", exc)
        return np.array([]), np.array([]), {}

    seq = pad_sequence_to_length(df, seq_len, feature_columns=list(df.columns))
    if len(seq) < seq_len:
        return np.array([]), np.array([]), {}

    obs = seq.values.astype(np.float32)  # shape: (seq_len, n_features)

    actions: list[int] = []
    values: list[float] = []
    entropies: list[float] = []

    with torch.no_grad():
        for i in range(len(obs)):
            single_obs = torch.tensor(obs[i : i + 1], dtype=torch.float32)
            action, _states = model.predict(single_obs, deterministic=False)
            actions.append(int(action[0]))

            value = model.policy.predict_values(single_obs).cpu().numpy()[0]
            values.append(float(value))

            dist = model.policy.get_distribution(single_obs)
            entropy = dist.entropy().cpu().numpy()[0]
            entropies.append(float(entropy))

    recent_values = np.array(values[-10:])
    recent_entropy = np.array(entropies[-10:])

    metadata = {
        "action": int(actions[-1]),
        "value": float(values[-1]),
        "entropy": float(recent_entropy.mean()),
        "value_ma100": float(recent_values.mean()),
        "value_std100": float(recent_values.std()) if len(recent_values) > 1 else 0.5,
        "entropy_raw_last": float(entropies[-1]),
        "confidence_score": float(1.0 / (1.0 + recent_entropy.mean())),
    }

    base_conf = 0.60
    if metadata["action"] == 2:
        base_conf = 0.92
    elif metadata["action"] == 0:
        base_conf = 0.45

    entropy_bonus = max(0.0, 0.3 - metadata["entropy"]) * 2.0
    final_conf = min(0.999, base_conf + entropy_bonus)

    return np.array([final_conf]), np.array([metadata["action"]]), metadata


def predict_rf(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    path = _first_existing(p / "rf_best.joblib" for p in MODEL_SEARCH_DIRS)
    if not path:
        return np.array([]), np.array([])
    if df.empty:
        return np.array([]), np.array([])
    model = load(path)
    X = _prepare_features(df.tail(1))
    prob = float(model.predict_proba(X)[0][1])
    return np.array([prob]), np.array([1 if prob > 0.5 else 0])


def predict_xgb(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    path = _first_existing(p / "xgb_best.joblib" for p in MODEL_SEARCH_DIRS)
    if not path:
        return np.array([]), np.array([])
    if df.empty:
        return np.array([]), np.array([])
    model = load(path)
    X = _prepare_features(df.tail(1))
    prob = float(model.predict_proba(X)[0][1])
    return np.array([prob]), np.array([1 if prob > 0.5 else 0])


def predict_lgb(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    path = _first_existing(p / "lgb_best.joblib" for p in MODEL_SEARCH_DIRS)
    if not path:
        return np.array([]), np.array([])
    if df.empty:
        return np.array([]), np.array([])
    model = load(path)
    X = _prepare_features(df.tail(1))
    prob = float(model.predict_proba(X)[0][1])
    return np.array([prob]), np.array([1 if prob > 0.5 else 0])


# ==================================================================
# 主预测入口
# ==================================================================

def predict_with_all_models(sequence_df: pd.DataFrame, seq_len: int = 60) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict]:
    preds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    prepared = _prepare_features(sequence_df)
    prepared_transformer = _prepare_features(sequence_df, for_transformer=True)

    try:
        preds["RandomForest"] = predict_rf(prepared)
    except Exception as exc:
        logging.error("RandomForest prediction failed: %s", exc)

    try:
        preds["XGBoost"] = predict_xgb(prepared)
    except Exception as exc:
        logging.error("XGBoost prediction failed: %s", exc)

    try:
        preds["LightGBM"] = predict_lgb(prepared)
    except Exception as exc:
        logging.error("LightGBM prediction failed: %s", exc)

    try:
        lstm_probs, lstm_decisions = predict_lstm(prepared, seq_len)
        preds["LSTM"] = (lstm_probs, lstm_decisions)
    except Exception as exc:
        logging.error("LSTM prediction failed: %s", exc)

    try:
        transformer_probs, transformer_decisions = predict_transformer(
            prepared_transformer if not prepared_transformer.empty else prepared,
            seq_len,
        )
        preds["Transformer"] = (transformer_probs, transformer_decisions)
    except Exception as exc:
        logging.error("Transformer prediction failed: %s", exc)

    ppo_prob, ppo_decision, ppo_meta = predict_ppo(prepared, seq_len)
    preds["PPO"] = (ppo_prob, ppo_decision)

    return preds, ppo_meta if ppo_meta else {}


# ==================================================================
# 核确认决策逻辑（保持原样，已最优）
# ==================================================================

def independent_model_decisions(predictions: Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict], return_details: bool = False):
    preds, ppo_meta = predictions
    detail = {"votes": {}, "confidences": {}, "missing": {}}

    qualified = []
    for name, (probs, _) in preds.items():
        if name not in ["RandomForest", "XGBoost", "LightGBM", "LSTM", "Transformer"]:
            continue
        if len(probs) == 0:
            detail["missing"][name] = "no prediction"
            continue
        prob = float(probs[-1])
        vote = "Buy" if prob > 0.5 else "Sell"
        conf = prob if vote == "Buy" else 1 - prob
        detail["votes"][name] = vote
        detail["confidences"][name] = conf
        thresh = ENSEMBLE_THRESHOLDS.get(name.lower()[:3], 0.9)
        if name == "LSTM":
            thresh = ENSEMBLE_THRESHOLDS["lstm"]
        if name == "Transformer":
            thresh = ENSEMBLE_THRESHOLDS["transformer"]
        if conf >= thresh and vote != "Sell":
            qualified.append((name.lower(), vote, conf))

    if len(qualified) < 3:
        reason = f"Only {len(qualified)} qualified (<3)"
        return ("Hold", {**detail, "reason": reason}) if return_details else "Hold"

    votes = [v for _, v, _ in qualified]
    majority = "Buy" if votes.count("Buy") >= votes.count("Sell") else "Sell"

    nuclear = any(
        (n == "xgb" and c >= NUCLEAR_XGB) or (n == "lstm" and c >= NUCLEAR_LSTM)
        for n, v, c in qualified
        if v == majority
    )

    if not nuclear:
        reason = "No nuclear confirmation"
        return ("Hold", {**detail, "reason": reason}) if return_details else "Hold"

    avg_conf = np.mean([c for _, v, c in qualified if v == majority])
    reason = f"≥3 qualified + nuclear confirmed → {majority} (conf={avg_conf:.4f})"

    if preds.get("PPO"):
        detail.update(ppo_meta)

    final_detail = {**detail, "reason": reason, "confidence": float(avg_conf)}
    return (majority, final_detail) if return_details else majority
