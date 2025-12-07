#!/usr/bin/env python3
"""
Ensemble backtest: combine multiple models' decisions to find the
optimal entry/exit strategy for maximum return.

Usage:
    python analyze_ensemble_backtest.py \
        --csv /app/data/trade_log_backtest_all_tickers.csv

You can edit:
- ENSEMBLES      : how models are combined (which models, how many must agree)
- EXIT_STYLES    : which exit styles to evaluate (A–E)
- TP/SL          : take-profit / stop-loss values for C/E
- CONF_THRESHOLDS: confidence levels to test (here from 1.0 down to 0.5)
"""

import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ----------------------------- Config --------------------------------- #

# Exit styles:
# A: fixed 4h exit
# B: fixed 8h exit
# C: TP/SL only (with max hold)
# D: reverse-signal only
# E: hybrid (TP/SL + reverse + max hold)
EXIT_STYLES = ["A", "B", "C", "D", "E"]

# Confidence thresholds to test (applied to every model in the ensemble)
# From 1.0 down to 0.5, step 0.01
CONF_THRESHOLDS = [round(th, 2) for th in np.linspace(1.0, 0.5, 51)]

# Take profit / stop loss used in styles C and E (in return space)
TP_DEFAULT = 0.02   # +2%
SL_DEFAULT = 0.01   # -1%
MAX_HOURS_DEFAULT = 24


@dataclass
class EnsembleConfig:
    name: str
    models: List[str]  # list of model names, e.g. ["lstm", "xgb"]
    k: int             # min number of agreeing models needed to fire a signal


# Some reasonable ensemble configurations to start with
ENSEMBLES: List[EnsembleConfig] = [
    EnsembleConfig(name="lstm_only", models=["lstm"], k=1),
    EnsembleConfig(name="xgb_only", models=["xgb"], k=1),
    EnsembleConfig(name="lstm_xgb_agree", models=["lstm", "xgb"], k=2),
    EnsembleConfig(
        name="lstm_xgb_trans_2of3",
        models=["lstm", "xgb", "transformer"],
        k=2,
    ),
    EnsembleConfig(
        name="lstm_xgb_trans_3of3",
        models=["lstm", "xgb", "transformer"],
        k=3,
    ),
    EnsembleConfig(
        name="all_models_majority",
        models=["rf", "xgb", "lgb", "lstm", "tcn", "ppo", "transformer"],
        k=4,  # majority of 7
    ),
]


# -------------------------- Core Functions ----------------------------- #

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    # Normalize vote strings
    vote_cols = [c for c in df.columns if c.endswith("_vote")]
    for c in vote_cols:
        df[c] = df[c].astype(str).str.upper()

    # Confidence to float
    conf_cols = [c for c in df.columns if c.endswith("_conf")]
    for c in conf_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def compute_ensemble_signal(
    row: pd.Series,
    ensemble: EnsembleConfig,
    conf_thresh: float,
) -> str:
    """
    For a single row, compute ensemble decision:

      - Consider only models in ensemble.models
      - For each model:
          if vote in {BUY, SELL} and conf >= conf_thresh
      - Count BUY signals and SELL signals
      - Fire:
          BUY  if buy_count >= k and buy_count >= sell_count
          SELL if sell_count >= k and sell_count >  buy_count
      - Return "BUY", "SELL", or "HOLD" (no trade)
    """
    buy_count = 0
    sell_count = 0

    for m in ensemble.models:
        vote_col = f"{m}_vote"
        conf_col = f"{m}_conf"
        if vote_col not in row or conf_col not in row:
            continue

        vote = row[vote_col]
        conf = row[conf_col]

        if vote not in ("BUY", "SELL"):
            continue
        if pd.isna(conf) or conf < conf_thresh:
            continue

        if vote == "BUY":
            buy_count += 1
        elif vote == "SELL":
            sell_count += 1

    if buy_count >= ensemble.k and buy_count >= sell_count and buy_count > 0:
        return "BUY"
    if sell_count >= ensemble.k and sell_count > buy_count and sell_count > 0:
        return "SELL"
    return "HOLD"


def backtest_ensemble(
    df: pd.DataFrame,
    ensemble: EnsembleConfig,
    conf_thresh: float,
    exit_style: str,
    tp: float = TP_DEFAULT,
    sl: float = SL_DEFAULT,
    max_hours: float = MAX_HOURS_DEFAULT,
    h4: float = 4.0,
    h8: float = 8.0,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Run backtest for a given ensemble config + confidence threshold + exit style.

    Returns:
        summary dict + trades DataFrame
    """
    open_pos = {}  # per ticker: {t0, p0, dir}
    trades = []

    # Group by ticker to avoid mixing instruments
    for ticker, g in df.groupby("ticker"):
        g = g.reset_index(drop=True)
        pos = None  # position state for this ticker only

        for _, row in g.iterrows():
            t = row["timestamp"]
            price = row["price"]

            # 1) Compute ensemble decision at this bar
            decision = compute_ensemble_signal(row, ensemble, conf_thresh)

            # 2) Manage existing position
            if pos is not None:
                dt_hours = (t - pos["t0"]).total_seconds() / 3600.0
                ret = (price - pos["p0"]) / pos["p0"]
                if pos["dir"] == "SELL":
                    ret = -ret

                exit_now = False
                reason = None

                if exit_style in ("A", "B"):
                    H = h4 if exit_style == "A" else h8
                    if dt_hours >= H:
                        exit_now = True
                        reason = f"time_{H}h"

                elif exit_style in ("C", "E"):
                    if ret >= tp:
                        exit_now = True
                        reason = "tp"
                    elif ret <= -sl:
                        exit_now = True
                        reason = "sl"

                    if not exit_now and dt_hours >= max_hours:
                        exit_now = True
                        reason = f"max_{max_hours}h"

                if exit_style in ("D", "E") and not exit_now:
                    # reverse-signal exit
                    if decision in ("BUY", "SELL") and decision != pos["dir"]:
                        exit_now = True
                        reason = "reverse"

                if exit_now:
                    trades.append(
                        {
                            "ensemble": ensemble.name,
                            "ticker": ticker,
                            "entry_time": pos["t0"],
                            "exit_time": t,
                            "direction": pos["dir"],
                            "entry_price": pos["p0"],
                            "exit_price": price,
                            "ret": ret,
                            "exit_reason": reason,
                            "conf_thresh": conf_thresh,
                            "exit_style": exit_style,
                        }
                    )
                    pos = None

            # 3) Enter new position if flat
            if pos is None and decision in ("BUY", "SELL"):
                pos = {"t0": t, "p0": price, "dir": decision}

        # 4) At end of ticker data, close any open position
        if pos is not None:
            price = g["price"].iloc[-1]
            t_last = g["timestamp"].iloc[-1]
            ret = (price - pos["p0"]) / pos["p0"]
            if pos["dir"] == "SELL":
                ret = -ret
            trades.append(
                {
                    "ensemble": ensemble.name,
                    "ticker": ticker,
                    "entry_time": pos["t0"],
                    "exit_time": t_last,
                    "direction": pos["dir"],
                    "entry_price": pos["p0"],
                    "exit_price": price,
                    "ret": ret,
                    "exit_reason": "eod",
                    "conf_thresh": conf_thresh,
                    "exit_style": exit_style,
                }
            )

    tr_df = pd.DataFrame(trades)
    if tr_df.empty:
        summary = {
            "ensemble": ensemble.name,
            "conf_thresh": conf_thresh,
            "exit_style": exit_style,
            "n": 0,
            "tot": 0.0,
            "avg": 0.0,
            "winrate": 0.0,
        }
        return summary, tr_df

    n_trades = len(tr_df)
    total_return = tr_df["ret"].sum()
    avg_return = tr_df["ret"].mean()
    winrate = (tr_df["ret"] > 0).mean()

    summary = {
        "ensemble": ensemble.name,
        "conf_thresh": conf_thresh,
        "exit_style": exit_style,
        "n": n_trades,
        "tot": total_return,
        "avg": avg_return,
        "winrate": winrate,
    }
    return summary, tr_df


# --------------------------- Main Script ------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Search for optimal ensemble of models for maximum return."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to trade_log_backtest_all_tickers.csv",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of best strategies to print (default: 20)",
    )
    args = parser.parse_args()

    df = load_data(args.csv)

    results = []

    total_runs = len(ENSEMBLES) * len(CONF_THRESHOLDS) * len(EXIT_STYLES)
    with tqdm(total=total_runs, desc="Running backtests") as pbar:
        for ensemble in ENSEMBLES:
            for conf_thresh in CONF_THRESHOLDS:
                for style in EXIT_STYLES:
                    summary, _ = backtest_ensemble(
                        df,
                        ensemble,
                        conf_thresh,
                        exit_style=style,
                        tp=TP_DEFAULT,
                        sl=SL_DEFAULT,
                        max_hours=MAX_HOURS_DEFAULT,
                    )
                    results.append(summary)
                    pbar.update(1)

    res_df = pd.DataFrame(results)

    # Sort by total return descending
    res_df = res_df.sort_values("tot", ascending=False).reset_index(drop=True)

    print(res_df.head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
