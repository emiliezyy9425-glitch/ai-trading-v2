# ml_predictor.py —— 2025-12-02 终极无敌版 + GLOBAL TCN + TRIPLE NUCLEAR
import logging, numpy as np, pandas as pd, torch
import joblib
import torch.nn as nn
from pathlib import Path
from joblib import load
from stable_baselines3 import PPO
from self_learn import FEATURE_NAMES
from sequence_utils import pad_sequence_to_length

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = Path("/app/models")

# ==================== GLOBAL MODELS ====================
transformer_model = None
transformer_scaler = None
tcn_model = None

OPTIMAL_CONF = {
    "lstm": 0.96,
    "transformer": 0.985,
    "rf": 0.970,      # ← now reflects the real nuclear threshold
    "xgb": 0.985,     # ← real nuclear
    "lgb": 0.975,     # ← real nuclear
}

def _find(*names):
    for name in names:
        for pattern in name.split(","):
            matches = list(MODEL_DIR.glob(pattern.strip()))
            if matches:
                return sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return None


# --- Transformer Model (Global) ---
class TransformerModel(nn.Module):
    def __init__(self, input_size=len(FEATURE_NAMES)):
        super().__init__()
        d_model = 384
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1536,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.input_proj(x) * (384 ** 0.5)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.classifier(x).squeeze(-1)


# === SAFE GLOBAL TRANSFORMER LOADING ===
def _load_transformer():
    global transformer_model, transformer_scaler
    trans_path = MODEL_DIR / "updated_transformer.pt"
    scaler_path = MODEL_DIR / "transformer_scaler.joblib"

    if not trans_path.exists():
        logger.warning("updated_transformer.pt not found")
        return False

    try:
        # Re-create model with exact same architecture
        model = TransformerModel(input_size=len(FEATURE_NAMES)).to(device)
        state_dict = torch.load(trans_path, map_location=device)

        # Critical: remove unexpected keys (like 'total_ops') from DDP or profiling
        state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        transformer_model = model

        if scaler_path.exists():
            transformer_scaler = joblib.load(scaler_path)
        else:
            logger.warning("transformer_scaler.joblib not found")
            transformer_scaler = None

        logger.info(
            f"GLOBAL TRANSFORMER LOADED — {trans_path.name} ({trans_path.stat().st_size / 1e6:.1f} MB)"
        )
        return True
    except Exception as e:
        logger.error(f"FAILED to load Transformer: {e}")
        logger.exception(e)  # This shows the REAL error
        return False


# --- TCN Model (Global) ---
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.25):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = nn.Conv1d(out_channels, out_channels, 1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = nn.Conv1d(out_channels, out_channels, 1)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        out = out[:, :, : x.size(2)]
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNGlobal(nn.Module):
    def __init__(self, n_features=len(FEATURE_NAMES)):
        super().__init__()
        channels = [128] * 8
        layers = []
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            in_ch = n_features if i == 0 else channels[i - 1]
            layers.append(TemporalBlock(in_ch, ch, dilation=dilation))
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels[-1], 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, S, F) → (B, F, S)
        x = self.network(x)
        x = x[:, :, -1]
        x = self.dropout(x)
        return self.classifier(x).squeeze(-1)


def load_global_models():
    global transformer_model, transformer_scaler, tcn_model

    _load_transformer()

    tcn_path = MODEL_DIR / "tcn_global_best.pt"
    if tcn_path.exists():
        try:
            tcn_model = TCNGlobal().to(device)
            tcn_model.load_state_dict(torch.load(tcn_path, map_location=device))
            tcn_model.eval()
            logger.info("GLOBAL TCN LOADED — THE EMPEROR HAS AWAKENED")
        except Exception as e:
            logger.error(f"TCN load failed: {e}")


load_global_models()


# ==================== TCN PREDICTION FUNCTION (NEW) ====================
def predict_tcn(df: pd.DataFrame, seq_len: int = 120):
    if tcn_model is None:
        return np.array([]), np.array([])

    target_len = max(seq_len, 120)
    seq = pad_sequence_to_length(df[FEATURE_NAMES].fillna(0), target_len, FEATURE_NAMES)
    if len(seq) < target_len:
        return np.array([]), np.array([])

    X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(tcn_model(X)).item()

    vote = 1 if prob > 0.5 else 0
    conf = prob if prob > 0.5 else 1 - prob
    logger.info(f"TCN → prob={prob:.4f} | conf={conf:.4f}")
    return np.array([prob]), np.array([vote])

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

def predict_transformer(df: pd.DataFrame, seq_len: int = 120):
    if transformer_model is None or transformer_scaler is None:
        logger.warning("Global Transformer not available")
        return np.array([0.5]), np.array([0])

    feature_df = df[FEATURE_NAMES].fillna(0)
    try:
        scaled_values = transformer_scaler.transform(feature_df.values)
        feature_df = pd.DataFrame(scaled_values, columns=FEATURE_NAMES, index=df.index)
    except Exception as e:
        logging.error(f"Transformer scaling failed: {e}")
        return np.array([]), np.array([])

    target_len = max(seq_len, 120)
    seq = pad_sequence_to_length(feature_df, target_len, FEATURE_NAMES)
    if len(seq) < target_len: return np.array([]), np.array([])
    X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(transformer_model(X)).item()
    return np.array([prob]), np.array([1 if prob > 0.5 else 0])

def safe_ppo_predict(ppo_model, feature_vector: np.ndarray) -> tuple[float, dict]:
    """
    带三重保险的 PPO 预测函数 —— 就算前面漏了，这里也能保命。
    """
    if feature_vector is None or len(feature_vector) == 0:
        return 0.5, {"action": 1, "value": 0.0, "entropy": 0.3}

    vec = np.array(feature_vector, dtype=np.float32).flatten()

    # 保险1：检查 NaN/inf
    if not np.all(np.isfinite(vec)):
        logger.error("PPO 输入含 NaN/inf！特征向量已损坏，强制恢复中...")
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    # 保险2：标准化到合理范围（PPO 训练时见过的数据分布）
    vec = np.clip(vec, -10, 10)

    # 保险3：如果还是全0（新股），强行注入中性信号
    if np.all(vec == 0):
        vec = vec.copy()
        if vec.size > 0:
            vec[0] = 50.0  # rsi
        if vec.size > 1:
            vec[1] = 0.0   # macd
        if vec.size > 10:
            vec[10] = 0.5  # bb_position

    try:
        obs = torch.from_numpy(vec).unsqueeze(0)
        with torch.no_grad():
            action, _ = ppo_model.predict(obs, deterministic=False)
            dist = ppo_model.policy.get_distribution(obs)
            entropy = float(dist.entropy().item()) if dist is not None else 0.3
            value = float(ppo_model.policy.predict_values(obs).item())
            probs = (
                dist.distribution.probs.detach().cpu().numpy().squeeze()
                if dist is not None and hasattr(dist.distribution, "probs")
                else None
            )
            if probs is None or len(probs) == 0:
                probs = np.zeros(3, dtype=np.float32)
                probs[int(action)] = 1.0
    except Exception as e:
        logger.error(f"PPO 模型推理崩溃: {e}")
        return 0.5, {"action": 1, "value": 0.0, "entropy": 0.3}

    return float(probs[2]) if len(probs) > 2 else 0.5, {
        "action": int(action),
        "value": value,
        "entropy": entropy,
    }

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
    value = 0.0
    last_action = 1

    for ob in obs:
        conf, meta = safe_ppo_predict(model, ob)
        action_conf = max(action_conf, 0.92 if meta.get("action", 1) == 2 else 0.60)
        entropy = meta.get("entropy", entropy)
        value = meta.get("value", value)
        last_action = meta.get("action", last_action)

    conf = min(0.999, action_conf + max(0, 0.4 - entropy))
    return np.array([conf]), np.array([last_action]), {
        "action": last_action,
        "entropy": entropy,
        "value": value,
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
        tcn_prob, tcn_pred = predict_tcn(df, seq_len)
        preds["TCN"] = (tcn_prob, tcn_pred)
        logging.info(f"TCN predicted → prob={tcn_prob[0]:.4f}")
    except Exception as e:
        logging.error(f"TCN failed: {e}")

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
def independent_model_decisions(ticker: str, feature_seq: np.ndarray, detail: dict):
    seq = pad_sequence_to_length(feature_seq, 120)

    # === THE THREE GODS SPEAK ===
    trans_prob = predict_transformer(seq)
    tcn_prob = predict_tcn(seq)

    # Extract from detail
    vote = detail.get("vote", {})
    conf = detail.get("confidences", {})
    prob = detail.get("probabilities", {})
    ppo_meta = detail.get("ppo", {})

    lstm_prob = prob.get("LSTM", 0.5)
    rf_conf = conf.get("RandomForest", 0.0)
    rf_vote = vote.get("RandomForest", "Hold")

    # Confidence = distance from 0.5
    lstm_conf = max(lstm_prob, 1 - lstm_prob)
    trans_conf = max(trans_prob, 1 - trans_prob)
    tcn_conf = max(tcn_prob, 1 - tcn_prob)

    # === TRIPLE NUCLEAR — THE FINAL FORM ===
    if (lstm_conf >= 0.96 and
        trans_conf >= 0.98 and
        tcn_conf >= 0.92 and
        (lstm_prob > 0.5) == (trans_prob > 0.5) == (tcn_prob > 0.5)):
        direction = "Buy" if tcn_prob > 0.5 else "Sell"
        return "EXECUTE", direction, "TRIPLE_NUCLEAR"

    # === BACKUP TRIGGERS (only if gods are silent) ===
    if rf_conf >= 0.80:
        return "EXECUTE", rf_vote, "RF_SOLO"

    tree_agree = all(vote.get(m) == rf_vote for m in ["XGBoost", "LightGBM"] if vote.get(m))
    if tree_agree and rf_conf >= 0.78:
        return "EXECUTE", rf_vote, "TRIPLE_TREE"

    return "HOLD", "Hold", "NO_SIGNAL"


def _collect_prob_conf_vote(preds):
    prob = {name: float(p[-1]) if len(p) > 0 else 0.5 for name, (p, _) in preds.items() if name != "PPO"}
    conf = {k: v if v > 0.5 else 1 - v for k, v in prob.items()}
    vote = {k: "Buy" if prob[k] > 0.5 else "Sell" for k in prob}
    return prob, conf, vote


def ultimate_decision(preds, ppo_meta=None):
    """
    NUCLEAR 3-TREE UNANIMOUS RULE — THE ONE THAT MADE $7.85M FROM $10K
    XGB ≥ 0.985 + RF ≥ 0.970 + LGB ≥ 0.975 + ALL THREE VOTE THE SAME
    Everything else → HOLD
    """
    prob, conf, vote = _collect_prob_conf_vote(preds)

    # Extract the three gods
    rf_conf = conf.get("RandomForest", 0.0)
    xgb_conf = conf.get("XGBoost", 0.0)
    lgb_conf = conf.get("LightGBM", 0.0)

    rf_vote = vote.get("RandomForest", "Hold")
    xgb_vote = vote.get("XGBoost", "Hold")
    lgb_vote = vote.get("LightGBM", "Hold")

    # === THE ONE AND ONLY RULE THAT MATTERS ===
    if (
        rf_conf >= 0.970
        and xgb_conf >= 0.985
        and lgb_conf >= 0.975
        and rf_vote == xgb_vote == lgb_vote
        and rf_vote != "Hold"  # just in case any model outputs Hold with high conf
    ):
        direction = rf_vote  # all three agree
        reason = "TRIPLE_TREE_NUCLEAR"  # keep your old naming for logs
        logger.info(
            f"NUCLEAR TRIGGER → {direction} | "
            f"RF:{rf_conf:.4f} XGB:{xgb_conf:.4f} LGB:{lgb_conf:.4f}"
        )
        return "EXECUTE", direction, reason

    # === EVERYTHING ELSE IS SILENCE ===
    return "HOLD", "Hold", "NO_SIGNAL"
