#!/usr/bin/env python3
"""
Ensemble backtest: combine multiple models' decisions to find the
optimal entry/exit strategy for maximum return.

This version:
- Sweeps over:
    * ensembles
    * confidence thresholds
    * exit styles (A–E)
    * take profit (TP) and stop loss (SL) grids for styles C & E
- Shows a tqdm progress bar.
"""

import argparse
from dataclasses import dataclass
from typing import List, Dict

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

# Confidence sweep defaults (you can override via CLI)
CONF_MIN_DEFAULT = 0.90
CONF_MAX_DEFAULT = 1.00
CONF_STEP_DEFAULT = 0.01

# TP / SL grids for exit styles C & E
# Values are in "return space" (e.g. 0.02 = +2%)
TP_LEVELS = [0.01, 0.02, 0.03]   # +1%, +2%, +3%
SL_LEVELS = [0.005, 0.01, 0.015] # -0.5%, -1%, -1.5%

# Max holding time (in hours) used for C & E
MAX_HOURS_DEFAULT = 24.0


@dataclass
class EnsembleConfig:
    name: str
    models: List[str]
    k: int  # minimum agreeing models


ALL_ENSEMBLES: List[EnsembleConfig] = [
    EnsembleConfig("lstm_only", ["lstm"], 1),
    EnsembleConfig("xgb_only", ["xgb"], 1),
    EnsembleConfig("lstm_xgb_agree", ["lstm", "xgb"], 2),
    EnsembleConfig("lstm_xgb_trans_2of3", ["lstm", "xgb", "transformer"], 2),
    EnsembleConfig("lstm_xgb_trans_3of3", ["lstm", "xgb", "transformer"], 3),
    EnsembleConfig(
        "all_models_majority",
        ["rf", "xgb", "lgb", "lstm", "tcn", "ppo", "transformer"],
        4,  # majority of 7
    ),
]


# -------------------------- Data Loader ------------------------------ #

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    vote_cols = [c for c in df.columns if c.endswith("_vote")]
    for c in vote_cols:
        df[c] = df[c].astype(str).str.upper()

    conf_cols = [c for c in df.columns if c.endswith("_conf")]
    for c in conf_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# --------------------------- Core Logic ------------------------------ #

def compute_ensemble_signal(row, ensemble: EnsembleConfig, conf_thresh: float) -> str:
    """Compute BUY/SELL/HOLD for one row from an ensemble."""
    buy = 0
    sell = 0

    for m in ensemble.models:
        vote_col = f"{m}_vote"
        conf_col = f"{m}_conf"
        if vote_col not in row or conf_col not in row:
            continue

        v = row[vote_col]
        c = row[conf_col]

        if v not in ("BUY", "SELL"):
            continue
        if pd.isna(c) or c < conf_thresh:
            continue

        if v == "BUY":
            buy += 1
        elif v == "SELL":
            sell += 1

    if buy >= ensemble.k and buy >= sell and buy > 0:
        return "BUY"
    if sell >= ensemble.k and sell > buy and sell > 0:
        return "SELL"
    return "HOLD"


def backtest_ensemble(
    df: pd.DataFrame,
    ensemble: EnsembleConfig,
    conf_thresh: float,
    exit_style: str,
    tp: float = None,
    sl: float = None,
    max_hours: float = MAX_HOURS_DEFAULT,
) -> Dict:
    """
    Backtest a single (ensemble, conf_thresh, exit_style, tp, sl) combination.

    For exit_style:
      A: fixed 4h
      B: fixed 8h
      C: TP/SL only (plus max_hours)
      D: reverse only
      E: TP/SL + reverse + max_hours
    """
    trades = []

    for ticker, g in df.groupby("ticker"):
        g = g.reset_index(drop=True)
        pos = None  # {t0, p0, dir}

        for _, row in g.iterrows():
            t = row["timestamp"]
            price = row["price"]
            decision = compute_ensemble_signal(row, ensemble, conf_thresh)

            # Manage exit
            if pos is not None:
                dt_hours = (t - pos["t0"]).total_seconds() / 3600.0
                ret = (price - pos["p0"]) / pos["p0"]
                if pos["dir"] == "SELL":
                    ret = -ret

                exit_now = False
                reason = None

                # Time-based exits A/B
                if exit_style == "A" and dt_hours >= 4:
                    exit_now, reason = True, "4h"
                elif exit_style == "B" and dt_hours >= 8:
                    exit_now, reason = True, "8h"

                # TP/SL exits for C & E
                if exit_style in ("C", "E") and not exit_now:
                    if tp is not None and ret >= tp:
                        exit_now, reason = True, "tp"
                    elif sl is not None and ret <= -sl:
                        exit_now, reason = True, "sl"
                    elif dt_hours >= max_hours:
                        exit_now, reason = True, f"max_{max_hours}h"

                # Reverse-signal exits for D & E
                if exit_style in ("D", "E") and not exit_now:
                    if decision in ("BUY", "SELL") and decision != pos["dir"]:
                        exit_now, reason = True, "reverse"

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
                            "tp": tp,
                            "sl": sl,
                        }
                    )
                    pos = None

            # Manage entry
            if pos is None and decision in ("BUY", "SELL"):
                pos = {"t0": t, "p0": price, "dir": decision}

        # Close any open position at the end of that ticker's history
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
                    "tp": tp,
                    "sl": sl,
                }
            )

    tr = pd.DataFrame(trades)
    if tr.empty:
        return {
            "ensemble": ensemble.name,
            "conf_thresh": conf_thresh,
            "exit_style": exit_style,
            "tp": tp,
            "sl": sl,
            "n": 0,
            "tot": 0.0,
            "avg": 0.0,
            "win": 0.0,
        }

    total = tr["ret"].sum()
    avg = tr["ret"].mean()
    winrate = (tr["ret"] > 0).mean()

    return {
        "ensemble": ensemble.name,
        "conf_thresh": conf_thresh,
        "exit_style": exit_style,
        "tp": tp,
        "sl": sl,
        "n": len(tr),
        "tot": float(total),
        "avg": float(avg),
        "win": float(winrate),
    }


# --------------------------- Main Script ------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Ensemble backtest with TP/SL optimization."
    )
    parser.add_argument("--csv", required=True,
                        help="Path to trade_log_backtest_all_tickers.csv")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of best rows to print")
    parser.add_argument("--min-conf", type=float, default=CONF_MIN_DEFAULT,
                        help="Minimum confidence threshold (default 0.90)")
    parser.add_argument("--max-conf", type=float, default=CONF_MAX_DEFAULT,
                        help="Maximum confidence threshold (default 1.00)")
    parser.add_argument("--conf-step", type=float, default=CONF_STEP_DEFAULT,
                        help="Step size for confidence sweep (default 0.01)")
    parser.add_argument(
        "--ensembles",
        type=str,
        default="lstm_only,xgb_only,lstm_xgb_agree",
        help=("Comma-separated ensemble names to test. "
              "Available: lstm_only,xgb_only,lstm_xgb_agree,"
              "lstm_xgb_trans_2of3,lstm_xgb_trans_3of3,all_models_majority"),
    )

    args = parser.parse_args()

    df = load_data(args.csv)

    # Build confidence grid
    conf_vals = np.arange(args.max_conf, args.min_conf - 1e-9, -args.conf_step)
    conf_vals = [round(float(c), 4) for c in conf_vals]

    # Select ensembles
    requested = {e.strip() for e in args.ensembles.split(",") if e.strip()}
    ensembles = [e for e in ALL_ENSEMBLES if e.name in requested]
    if not ensembles:
        raise ValueError("No valid ensembles selected. Check --ensembles argument.")

    # Compute total jobs for progress bar
    num_conf = len(conf_vals)
    num_ensembles = len(ensembles)
    num_styles_no_tp = len([s for s in EXIT_STYLES if s not in ("C", "E")])
    num_styles_tp = len([s for s in EXIT_STYLES if s in ("C", "E")])

    jobs_no_tp = num_ensembles * num_conf * num_styles_no_tp
    jobs_tp = num_ensembles * num_conf * num_styles_tp * len(TP_LEVELS) * len(SL_LEVELS)
    total_jobs = jobs_no_tp + jobs_tp

    print(f"Ensembles: {[e.name for e in ensembles]}")
    print(f"Confidence grid: {conf_vals[0]} -> {conf_vals[-1]} (step={args.conf_step})")
    print(f"TP levels: {TP_LEVELS}")
    print(f"SL levels: {SL_LEVELS}")
    print(f"Total combinations to run: {total_jobs}")

    results = []

    with tqdm(total=total_jobs, desc="Ensemble+TP/SL Backtest", ncols=100) as bar:
        for ensemble in ensembles:
            for conf in conf_vals:
                # First, exit styles without TP/SL (A, B, D)
                for style in EXIT_STYLES:
                    if style not in ("C", "E"):
                        summary = backtest_ensemble(
                            df,
                            ensemble,
                            conf_thresh=conf,
                            exit_style=style,
                            tp=None,
                            sl=None,
                            max_hours=MAX_HOURS_DEFAULT,
                        )
                        results.append(summary)
                        bar.update(1)

                # Then styles that use TP/SL (C, E)
                for style in EXIT_STYLES:
                    if style in ("C", "E"):
                        for tp in TP_LEVELS:
                            for sl in SL_LEVELS:
                                summary = backtest_ensemble(
                                    df,
                                    ensemble,
                                    conf_thresh=conf,
                                    exit_style=style,
                                    tp=tp,
                                    sl=sl,
                                    max_hours=MAX_HOURS_DEFAULT,
                                )
                                results.append(summary)
                                bar.update(1)

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("tot", ascending=False).reset_index(drop=True)

    print("\n===== TOP STRATEGIES (including TP/SL) =====\n")
    print(res_df.head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
