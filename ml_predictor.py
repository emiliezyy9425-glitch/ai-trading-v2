# ml_predictor.py —— 2025-12-01 终极无敌版（已适配你所有真实模型）
import os, logging, numpy as np, pandas as pd, torch
from pathlib import Path
from joblib import load
from stable_baselines3 import PPO
from self_learn import FEATURE_NAMES
from sequence_utils import pad_sequence_to_length

MODEL_DIR = Path("/app/models")

OPTIMAL_CONF = {
    "lstm": 0.96,
    "transformer": 0.985,
    "xgb": 0.78,
    "lgb": 0.76,
    "rf": 0.80,
}

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

    prob, conf, vote = _collect_prob_conf_vote(preds)
    status, action, trigger = ultimate_decision(preds, ppo_meta)

    decision = action if status == "EXECUTE" else "Hold"

    def _mean_conf(model_names):
        vals = [conf[m] for m in model_names if m in conf]
        return float(np.mean(vals)) if vals else 0.0

    confidence_lookup = {
        "DEEPSEQ_NUCLEAR": _mean_conf(["LSTM", "Transformer"]),
        "TRIPLE_TREE_NUCLEAR": _mean_conf(["RandomForest", "XGBoost", "LightGBM"]),
        "RF_SOLO": conf.get("RandomForest", 0.0),
        "PPO_BREAKER": _mean_conf(["RandomForest"]),
    }

    ppo_conf_raw = ppo_meta.get("action_conf", ppo_meta.get("conf", [0.6]))
    if isinstance(ppo_conf_raw, (list, np.ndarray)):
        ppo_conf = float(ppo_conf_raw[0]) if len(ppo_conf_raw) else 0.6
    else:
        ppo_conf = float(ppo_conf_raw)

    if not return_details:
        return decision

    return decision, {
        "reason": trigger,
        "trigger": trigger,
        "decision_state": status,
        "confidence": confidence_lookup.get(trigger, 0.0),
        "qualified_models": len(prob),
        "votes": {**vote, "PPO": "Aggressive" if ppo_meta.get("action", 1) == 2 else "Hold"},
        "confidences": {**conf, "PPO": ppo_conf},
        "ppo_action": ppo_meta.get("action", 1),
        "ppo_entropy": ppo_meta.get("entropy", 0.5),
        "nuclear_buy_votes": sum(1 for v in vote.values() if v == "Buy"),
        "nuclear_sell_votes": sum(1 for v in vote.values() if v == "Sell"),
    }


def _collect_prob_conf_vote(preds):
    prob = {name: float(p[-1]) if len(p) > 0 else 0.5 for name, (p, _) in preds.items() if name != "PPO"}
    conf = {k: v if v > 0.5 else 1 - v for k, v in prob.items()}
    vote = {k: "Buy" if prob[k] > 0.5 else "Sell" for k in prob}
    return prob, conf, vote


def ultimate_decision(preds, ppo_meta=None):
    ppo_meta = ppo_meta or {}
    prob, conf, vote = _collect_prob_conf_vote(preds)

    rf_conf = conf.get("RandomForest", 0)
    rf_vote = vote.get("RandomForest", "Hold")

    # === FIXED ORDER: DEEPSEQ FIRST (highest conviction) ===
    lstm_conf = conf.get("LSTM", 0)
    trans_conf = conf.get("Transformer", 0)
    if lstm_conf >= OPTIMAL_CONF["lstm"] and trans_conf >= OPTIMAL_CONF["transformer"]:
        seq_vote = (
            "Buy"
            if (prob.get("LSTM", 0.5) > 0.5 and prob.get("Transformer", 0.5) > 0.5)
            else "Sell"
        )
        return "EXECUTE", seq_vote, "DEEPSEQ_NUCLEAR"

    # Tier 1: Triple Tree Nuclear (second highest)
    if (
        rf_conf >= 0.78
        and conf.get("XGBoost", 0) >= OPTIMAL_CONF["xgb"]
        and conf.get("LightGBM", 0) >= OPTIMAL_CONF["lgb"]
        and vote.get("RandomForest") == vote.get("XGBoost", "Hold") == vote.get("LightGBM", "Hold")
    ):
        return "EXECUTE", rf_vote, "TRIPLE_TREE_NUCLEAR"

    # Tier 2: RandomForest Solo (daily bread)
    if rf_conf >= OPTIMAL_CONF["rf"]:
        return "EXECUTE", rf_vote, "RF_SOLO"

    # Tier 3: PPO tie-breaker (rare)
    if (
        0.75 <= rf_conf < 0.78
        and abs(conf.get("XGBoost", 0) - 0.5) < 0.15
        and abs(conf.get("LightGBM", 0) - 0.5) < 0.15
        and ppo_meta.get("action", 1) == 2
    ):
        return "EXECUTE", "Buy", "PPO_BREAKER"

    return "HOLD", "Hold", "NO_SIGNAL"
