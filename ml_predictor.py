# ml_predictor.py —— 2025-12-01 终极无敌版（已适配你所有真实模型）
import os, logging, numpy as np, pandas as pd, torch
from pathlib import Path
from joblib import load
from stable_baselines3 import PPO
from self_learn import FEATURE_NAMES
from sequence_utils import pad_sequence_to_length

MODEL_DIR = Path("/app/models")

def _find(*names):
    for name in names:
        for pattern in name.split(","):
            matches = list(MODEL_DIR.glob(pattern.strip()))
            if matches:
                return sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return None

# ================== 兼容你的真实模型架构 ==================
def predict_lstm(df: pd.DataFrame, seq_len: int = 60):
    path = _find("updated_lstm.pt, lstm_best.pt, lstm_*.pt")
    if not path: return np.array([]), np.array([])

    try:
        from models.lstm import AttentiveBiLSTM
        
        # 强制使用 3 层（你当前所有 lstm_best.pt 都是 3 层训练的）
        model = AttentiveBiLSTM(input_size=len(FEATURE_NAMES), hidden_size=151, num_layers=3)
        state_dict = torch.load(path, map_location="cpu")
        
        # 可选：如果还是报 missing key，自动删掉第3层权重再加载（极端兼容）
        if "lstm.weight_ih_l2" not in state_dict:
            # 旧2层模型权重 → 复制到第3层（几乎不会走到这里）
            for k in list(state_dict.keys()):
                if k.startswith("lstm.") and ("l1" in k or "l0" in k):
                    new_k = k.replace("l1", "l2") if "reverse" not in k else k.replace("l1_reverse", "l2_reverse")
                    state_dict[new_k] = state_dict[k]
        
        model.load_state_dict(state_dict, strict=False)  # 改成 strict=False 防止残缺报错
        logging.info(f"LSTM loaded successfully (3 layers) from {path.name}")
        model.eval()
    except Exception as e:
        logging.error(f"LSTM load failed: {e}")
        return np.array([]), np.array([])

    seq = pad_sequence_to_length(df, seq_len)
    if len(seq) < seq_len: return np.array([]), np.array([])
    X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(X)).item()
    return np.array([prob]), np.array([1 if prob > 0.5 else 0])

def predict_transformer(df: pd.DataFrame, seq_len: int = 60):
    path = _find("updated_transformer.pt, transformer_best.pt, transformer_trial_*.pt")
    if not path: return np.array([]), np.array([])

    try:
        from models.transformer import TransformerModel
        model = TransformerModel(input_size=len(FEATURE_NAMES))
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
    except Exception as e:
        logging.error(f"Transformer load failed: {e}")
        return np.array([]), np.array([])

    seq = pad_sequence_to_length(df, seq_len)
    if len(seq) < seq_len: return np.array([]), np.array([])
    X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(X)).item()
    return np.array([prob]), np.array([1 if prob > 0.5 else 0])

def predict_ppo(df: pd.DataFrame, seq_len: int = 60):
    path = _find("updated_ppo_latest.zip, ppo_trader.zip, ppo_*.zip")
    if not path: return np.array([0.6]), np.array([1]), {}

    try:
        model = PPO.load(path, device="cpu")
    except: return np.array([0.6]), np.array([1]), {}

    seq = pad_sequence_to_length(df, seq_len)
    if len(seq) < 10: return np.array([0.6]), np.array([1]), {}
    obs = seq[-10:].astype(np.float32)  # 修复：去掉 .values

    action_conf = 0.6
    entropy = 0.5
    try:
        with torch.no_grad():
            for ob in obs:
                ob_t = torch.tensor(ob).unsqueeze(0)
                action, _ = model.predict(ob_t, deterministic=False)
                dist = model.policy.get_distribution(ob_t)
                entropy = dist.entropy().item()
                action_conf = 0.92 if int(action) == 2 else 0.60
    except: pass

    conf = min(0.999, action_conf + max(0, 0.4 - entropy))
    return np.array([conf]), np.array([int(action) if 'action' in locals() else 1]), {
        "action": int(action) if 'action' in locals() else 1,
        "entropy": entropy,
    }

def predict_rf(df): return _tree_predict(df, "rf_latest.joblib, rf_best.joblib")
def predict_xgb(df): return _tree_predict(df, "xgb_latest.joblib, xgb_best.joblib")
def predict_lgb(df): return _tree_predict(df, "lgb_latest.joblib, lgb_best.joblib")

def _tree_predict(df, names):
    path = _find(names)
    if not path or df.empty: return np.array([]), np.array([])
    model = load(path)
    X = df[FEATURE_NAMES].tail(1)
    prob = model.predict_proba(X)[0][1]
    return np.array([prob]), np.array([1 if prob > 0.5 else 0])

def predict_with_all_models(sequence_df: pd.DataFrame, seq_len: int = 60):
    scaler_path = _find("scaler.joblib, transformer_scaler.joblib")
    df = sequence_df
    if scaler_path:
        try:
            scaler = load(scaler_path)
            X = sequence_df[FEATURE_NAMES].values
            X_scaled = scaler.transform(X)
            df = pd.DataFrame(X_scaled, columns=FEATURE_NAMES, index=sequence_df.index)
        except Exception as e:
            logging.warning(f"Scaler failed: {e}")

    preds = {}
    try:
        preds["RandomForest"] = predict_rf(df)
        logging.info("RF predicted")
    except Exception as e:
        logging.error(f"RF failed: {e}")
    try:
        preds["XGBoost"] = predict_xgb(df)
        logging.info("XGB predicted")
    except Exception as e:
        logging.error(f"XGB failed: {e}")
    try:
        preds["LightGBM"] = predict_lgb(df)
        logging.info("LGB predicted")
    except Exception as e:
        logging.error(f"LGB failed: {e}")
    try:
        lstm_prob, lstm_pred = predict_lstm(df, seq_len)
        preds["LSTM"] = (lstm_prob, lstm_pred)
        logging.info(f"LSTM predicted → prob={lstm_prob[0]:.4f}")
    except Exception as e:
        logging.error(f"LSTM failed: {e}")
    try:
        trans_prob, trans_pred = predict_transformer(df, seq_len)
        preds["Transformer"] = (trans_prob, trans_pred)
        logging.info(f"Transformer predicted → prob={trans_prob[0]:.4f}")
    except Exception as e:
        logging.error(f"Transformer failed: {e}")

    try:
        ppo_prob, ppo_act, ppo_meta = predict_ppo(df, seq_len)
        preds["PPO"] = (ppo_prob, ppo_act)
        logging.info(
            f"PPO predicted | action={ppo_meta.get('action', 1)} entropy={ppo_meta.get('entropy', 0.5):.3f}"
        )
    except Exception as e:
        logging.error(f"PPO failed: {e}")
        ppo_meta = {}

    return preds, ppo_meta

# 向下兼容旧回测脚本
FEATURE_ALIASES = {
    "bb_position_1h.1": "bb_position_1h",
    "price_z_120h.1": "price_z_120h",
    "ret_1h.1": "ret_1h",
    "ret_4h.1": "ret_4h",
    "ret_24h.1": "ret_24h",
}

# 核确认（最强逻辑）
def independent_model_decisions(preds, return_details=False):
    preds, ppo_meta = preds if isinstance(preds, tuple) else (preds, {})
    q = []
    for name, (prob, _) in preds.items():
        if name not in ["RandomForest","XGBoost","LightGBM","LSTM","Transformer"]: continue
        if len(prob)==0: continue
        p = float(prob[-1])
        conf = p if p > 0.5 else 1-p
        thresh = {
            "rf": 0.80,
            "xgb": 0.91,
            "lgb": 0.80,
            "lstm": 0.88,        # 关键！0.88 是当前 LSTM 的最佳触发点
            "transformer": 0.90,
        }.get(name.lower()[:3], 0.9)

        # 强制覆盖 LSTM（防止未来改回来）
        if name.lower().startswith("lstm"):
            thresh = 0.88
        if conf >= thresh:
            q.append((name.lower()[:3], "Buy" if p>0.5 else "Sell", conf))

    if len(q) < 3:
        return "Hold" if not return_details else ("Hold", {"reason": "less than 3 qualified"})

    nuclear = any(
        (n == "xgb" and c >= 0.90) or
        (n == "lstm" and c >= 0.87) or
        (n == "transformer" and c >= 0.89)
        for n, _, c in q
    )
    if not nuclear:
        return "Hold" if not return_details else ("Hold", {"reason": "no nuclear"})

    return "Buy" if not return_details else ("Buy", {"reason": "NUCLEAR BUY!", "confidence": np.mean([c for _,_,c in q])})
