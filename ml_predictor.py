# ml_predictor.py
# 2025-12-01 终极兼容版 —— 自动识别 updated_xxx / latest / trial 等真实文件名
# 支持你的全部模型：updated_lstm.pt, updated_transformer.pt, *_latest.joblib, updated_ppo_latest.zip

import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from joblib import load
from stable_baselines3 import PPO

os.environ["SB3_PPO_WARN"] = "0"

from self_learn import FEATURE_NAMES
from sequence_utils import pad_sequence_to_length

# ------------------------------------------------------------------
# 自动搜索模型目录
# ------------------------------------------------------------------
MODEL_DIRS = [Path("/app/models"), Path(__file__).parent / "models"]
MODEL_DIR = next((p for p in MODEL_DIRS if p.exists()), Path("/app/models"))

def _find_model(*candidates):
    """按优先级查找文件"""
    for pattern in candidates:
        matches = list(MODEL_DIR.glob(pattern))
        if matches:
            return sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return None

# ------------------------------------------------------------------
# 深度模型预测函数（自动匹配真实文件名）
# ------------------------------------------------------------------

def predict_lstm(df: pd.DataFrame, seq_len: int = 60):
    model_path = _find_model("updated_lstm.pt", "lstm_best.pt", "lstm_*.pt")
    if not model_path:
        logging.warning("LSTM model not found")
        return np.array([]), np.array([])

    try:
        from models.lstm import AttentiveBiLSTM
        model = AttentiveBiLSTM(input_size=len(FEATURE_NAMES))
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
    except Exception as e:
        logging.error(f"LSTM load failed: {e}")
        return np.array([]), np.array([])

    seq = pad_sequence_to_length(df, seq_len)
    if len(seq) < seq_len:
        return np.array([]), np.array([])

    X = torch.tensor(seq.values, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logit = model(X).item()
        prob = 1 / (1 + np.exp(-logit))
    return np.array([prob]), np.array([1 if prob > 0.5 else 0])


def predict_transformer(df: pd.DataFrame, seq_len: int = 60):
    model_path = _find_model("updated_transformer.pt", "transformer_best.pt", "transformer_trial_*.pt", "transformer_*.pt")
    if not model_path:
        logging.warning("Transformer model not found")
        return np.array([]), np.array([])

    try:
        from models.transformer import TransformerModel
        model = TransformerModel(input_size=len(FEATURE_NAMES))
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
    except Exception as e:
        logging.error(f"Transformer load failed: {e}")
        return np.array([]), np.array([])

    seq = pad_sequence_to_length(df, seq_len)
    if len(seq) < seq_len:
        return np.array([]), np.array([])

    X = torch.tensor(seq.values, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logit = model(X).item()
        prob = 1 / (1 + np.exp(-logit))
    return np.array([prob]), np.array([1 if prob > 0.5 else 0])


def predict_ppo(df: pd.DataFrame, seq_len: int = 60):
    model_path = _find_model("updated_ppo_latest.zip", "ppo_trader.zip", "ppo_latest.zip", "ppo_*.zip")
    if not model_path:
        logging.warning("PPO model not found")
        return np.array([0.6]), np.array([1]), {}

    try:
        model = PPO.load(model_path, device="cpu")
    except Exception as e:
        logging.error(f"PPO load failed: {e}")
        return np.array([0.6]), np.array([1]), {}

    seq = pad_sequence_to_length(df, seq_len)
    obs = seq.values.astype(np.float32)[-10:]  # PPO 只看最近10步

    actions, values, entropies = [], [], []
    with torch.no_grad():
        for i in range(len(obs)):
            ob = torch.tensor(obs[i:i+1])
            action, _ = model.predict(ob, deterministic=False)
            value = model.policy.predict_values(ob).item()
            entropy = model.policy.get_distribution(ob).entropy().item()
            actions.append(int(action))
            values.append(value)
            entropies.append(entropy)

    metadata = {
        "action": int(actions[-1]),
        "value": float(values[-1]),
        "entropy": float(np.mean(entropies[-5:])),
        "value_ma100": float(np.mean(values)),
        "value_std100": float(np.std(values)) if len(values)>1 else 0.5,
    }

    conf = 0.92 if metadata["action"] == 2 else 0.60 if metadata["action"] == 1 else 0.40
    conf += max(0, 0.3 - metadata["entropy"]) * 2
    return np.array([min(0.999, conf)]), np.array([metadata["action"]]), metadata


def predict_rf(df: pd.DataFrame):
    path = _find_model("rf_latest.joblib", "rf_best.joblib")
    if not path: return np.array([]), np.array([])
    model = load(path)
    X = df[FEATURE_NAMES].tail(1)
    prob = model.predict_proba(X)[0][1]
    return np.array([prob]), np.array([1 if prob > 0.5 else 0])


def predict_xgb(df: pd.DataFrame):
    path = _find_model("xgb_latest.joblib", "xgb_best.joblib")
    if not path: return np.array([]), np.array([])
    model = load(path)
    X = df[FEATURE_NAMES].tail(1)
    prob = model.predict_proba(X)[0][1]
    return np.array([prob]), np.array([1 if prob > 0.5 else 0])


def predict_lgb(df: pd.DataFrame):
    path = _find_model("lgb_latest.joblib", "lgb_best.joblib")
    if not path: return np.array([]), np.array([])
    model = load(path)
    X = df[FEATURE_NAMES].tail(1)
    prob = model.predict_proba(X)[0][1]
    return np.array([prob]), np.array([1 if prob > 0.5 else 0])


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------

def predict_with_all_models(sequence_df: pd.DataFrame, seq_len: int = 60):
    preds = {}
    ppo_meta = {}

    # 自动缩放（优先用你已有的 scaler）
    scaler_path = _find_model("scaler.joblib", "transformer_scaler.joblib", "ppo_scaler.joblib")
    if scaler_path:
        try:
            scaler = load(scaler_path)
            X = scaler.transform(sequence_df[FEATURE_NAMES])
            df_scaled = pd.DataFrame(X, columns=FEATURE_NAMES, index=sequence_df.index)
        except:
            df_scaled = sequence_df
    else:
        df_scaled = sequence_df

    try: preds["RandomForest"] = predict_rf(df_scaled)
    except: pass
    try: preds["XGBoost"] = predict_xgb(df_scaled)
    except: pass
    try: preds["LightGBM"] = predict_lgb(df_scaled)
    except: pass
    try: preds["LSTM"] = predict_lstm(df_scaled, seq_len)
    except: pass
    try: preds["Transformer"] = predict_transformer(df_scaled, seq_len)
    except: pass

    ppo_prob, ppo_act, ppo_meta = predict_ppo(df_scaled, seq_len)
    preds["PPO"] = (ppo_prob, ppo_act)

    return preds, ppo_meta


# ------------------------------------------------------------------
# 核确认决策（保持最强逻辑）
# ------------------------------------------------------------------

ENSEMBLE_THRESHOLDS = {"rf":0.825, "xgb":0.945, "lgb":0.825, "lstm":0.99, "transformer":0.985}
NUCLEAR_XGB, NUCLEAR_LSTM = 0.940, 0.990

def independent_model_decisions(predictions, return_details=False):
    preds, ppo_meta = predictions if isinstance(predictions, tuple) else (predictions, {})
    qualified = []

    for name, (probs, _) in preds.items():
        if name not in ["RandomForest","XGBoost","LightGBM","LSTM","Transformer"]:
            continue
        if len(probs) == 0: continue
        prob = float(probs[-1])
        vote = "Buy" if prob > 0.5 else "Sell"
        conf = prob if vote == "Buy" else 1 - prob
        thresh = ENSEMBLE_THRESHOLDS.get(name.lower()[:3], 0.9)
        if name == "LSTM": thresh = 0.99
        if name == "Transformer": thresh = 0.985
        if conf >= thresh:
            qualified.append((name.lower(), vote, conf))

    if len(qualified) < 3:
        reason = f"Only {len(qualified)} qualified"
        return ("Hold", {"reason": reason}) if return_details else "Hold"

    votes = [v for _,v,_ in qualified]
    majority = "Buy" if votes.count("Buy") > votes.count("Sell") else "Sell"

    nuclear = any(
        (n=="xgb" and c>=NUCLEAR_XGB) or (n=="lstm" and c>=NUCLEAR_LSTM)
        for n,v,c in qualified if v == majority
    )
    if not nuclear:
        return ("Hold", {"reason": "No nuclear"}) if return_details else "Hold"

    detail = {"reason": f"NUCLEAR {majority}!", "confidence": np.mean([c for _,v,c in qualified if v==majority])}
    if ppo_meta: detail.update(ppo_meta)
    return (majority, detail) if return_details else majority
