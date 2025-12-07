#!/usr/bin/env python3
"""
FULL AUTO-OPTIMIZER FOR MODEL-ENSEMBLE TRADING STRATEGIES
(Updated to support CLI early-stopping options)
"""

import argparse
import numpy as np
import pandas as pd
from itertools import combinations, product
from tqdm import tqdm
import math
import json

# ============================= CONFIG ================================= #

BASE_MODELS = ["rf", "xgb", "lgb", "lstm", "tcn", "ppo", "transformer"]

# Per-model thresholds: 0.50 → 1.00 (step 0.01)
CONF_GRID = [round(0.50 + 0.01 * i, 2) for i in range(51)]

EXIT_STYLES = ["A", "B", "C", "D", "E"]

# TP sweep: 1% → 10%
TP_LEVELS = [round(0.01 * i, 3) for i in range(1, 11)]

# SL sweep: 0.5% → 5.0%
SL_LEVELS = [round(0.005 * i, 3) for i in range(1, 11)]

MAX_HOURS_DEFAULT = 24.0


# ========================== DATA LOADING =============================== #

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    for c in df.columns:
        if c.endswith("_vote"):
            df[c] = df[c].astype(str).str.upper()
        if c.endswith("_conf"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ========================= SIGNAL FUNCTIONS =========================== #

def model_signal(row, model, thresh):
    v = row[f"{model}_vote"]
    c = row[f"{model}_conf"]
    if v not in ("BUY", "SELL"):
        return None
    if c is None or c < thresh:
        return None
    return v


def ensemble_vote(row, models, thresh_dict, k):
    votes = []
    for m in models:
        sig = model_signal(row, m, thresh_dict[m])
        if sig:
            votes.append(sig)

    if len(votes) < k:
        return "HOLD"

    buys = votes.count("BUY")
    sells = votes.count("SELL")

    if buys >= k and buys >= sells:
        return "BUY"
    if sells >= k and sells > buys:
        return "SELL"
    return "HOLD"


# =========================== BACKTEST ENGINE =========================== #

def backtest_strategy(
    df, models, conf_dict, k, exit_style,
    tp=None, sl=None,
    max_hours=MAX_HOURS_DEFAULT,
    early_stop=False,
    early_trades=250,
    early_threshold=-1.0,
):
    trades = []
    cum_ret = 0.0
    n_tr = 0
    kill = False

    for ticker, g in df.groupby("ticker"):
        if kill:
            break

        g = g.reset_index(drop=True)
        pos = None

        for _, row in g.iterrows():
            t = row["timestamp"]
            price = row["price"]

            vote = ensemble_vote(row, models, conf_dict, k)

            # ==== EXIT LOGIC ====
            if pos is not None:

                dt = (t - pos["t0"]).total_seconds() / 3600.0
                ret = (price - pos["p0"]) / pos["p0"]
                if pos["dir"] == "SELL":
                    ret = -ret

                exit_now, reason = False, None

                if exit_style == "A" and dt >= 4:
                    exit_now, reason = True, "4h"
                elif exit_style == "B" and dt >= 8:
                    exit_now, reason = True, "8h"

                if exit_style in ("C", "E") and not exit_now:
                    if tp and ret >= tp:
                        exit_now, reason = True, "tp"
                    elif sl and ret <= -sl:
                        exit_now, reason = True, "sl"
                    elif dt >= max_hours:
                        exit_now, reason = True, "maxH"

                if exit_style in ("D", "E") and not exit_now:
                    if vote in ("BUY", "SELL") and vote != pos["dir"]:
                        exit_now, reason = True, "reverse"

                if exit_now:
                    trades.append({
                        "ticker": ticker,
                        "entry_time": pos["t0"],
                        "exit_time": t,
                        "entry_price": pos["p0"],
                        "exit_price": price,
                        "direction": pos["dir"],
                        "ret": ret,
                        "exit_reason": reason,
                    })
                    pos = None

                    cum_ret += ret
                    n_tr += 1

                    if early_stop and n_tr >= early_trades and cum_ret <= early_threshold:
                        kill = True
                        break

            if kill:
                break

            # ==== ENTRY ====
            if pos is None and vote in ("BUY", "SELL"):
                pos = {"t0": t, "p0": price, "dir": vote}

        # close at end of last bar
        if pos is not None and not kill:
            price = g["price"].iloc[-1]
            t_last = g["timestamp"].iloc[-1]
            ret = (price - pos["p0"]) / pos["p0"]
            if pos["dir"] == "SELL":
                ret = -ret

            trades.append({
                "ticker": ticker,
                "entry_time": pos["t0"],
                "exit_time": t_last,
                "entry_price": pos["p0"],
                "exit_price": price,
                "direction": pos["dir"],
                "ret": ret,
                "exit_reason": "eod",
            })

            cum_ret += ret
            n_tr += 1

    if not trades:
        return {
            "models": models,
            "conf": conf_dict,
            "exit": exit_style,
            "tp": tp,
            "sl": sl,
            "n": 0,
            "tot": 0,
            "avg": 0,
            "win": 0,
        }

    tr = pd.DataFrame(trades)
    return {
        "models": models,
        "conf": conf_dict,
        "exit": exit_style,
        "tp": tp,
        "sl": sl,
        "n": len(tr),
        "tot": tr["ret"].sum(),
        "avg": tr["ret"].mean(),
        "win": (tr["ret"] > 0).mean(),
    }


# ===================== ENSEMBLES ============================ #

def generate_all_model_combinations():
    out = []
    for r in range(1, len(BASE_MODELS) + 1):
        for combo in combinations(BASE_MODELS, r):
            models = list(combo)
            k = math.ceil(len(models) / 2)
            out.append((models, k))
    return out


# ========================== OPTIMIZER =============================== #

def optimize(df, early_stop, early_trades, early_threshold, top_k=20):

    ensembles = generate_all_model_combinations()

    # ALL per-model thresholds
    strat_queue = []
    for models, k in ensembles:
        grids = [CONF_GRID for _ in models]
        for conf_vec in product(*grids):
            conf_dict = {models[i]: conf_vec[i] for i in range(len(models))}
            strat_queue.append((models, conf_dict, k))

    total_jobs = 0
    for (models, conf_dict, k) in strat_queue:
        for style in EXIT_STYLES:
            if style in ("C", "E"):
                total_jobs += len(TP_LEVELS) * len(SL_LEVELS)
            else:
                total_jobs += 1

    print(f"\nTotal strategies to evaluate: {total_jobs:,}\n")

    results = []

    with tqdm(total=total_jobs, desc="Optimizing", ncols=100) as bar:
        for (models, conf_dict, k) in strat_queue:
            for style in EXIT_STYLES:

                if style in ("C", "E"):
                    for tp, sl in product(TP_LEVELS, SL_LEVELS):
                        res = backtest_strategy(
                            df, models, conf_dict, k,
                            exit_style=style,
                            tp=tp, sl=sl,
                            early_stop=early_stop,
                            early_trades=early_trades,
                            early_threshold=early_threshold
                        )
                        results.append(res)
                        bar.update(1)

                else:
                    res = backtest_strategy(
                        df, models, conf_dict, k,
                        exit_style=style,
                        early_stop=early_stop,
                        early_trades=early_trades,
                        early_threshold=early_threshold
                    )
                    results.append(res)
                    bar.update(1)

    ranked = sorted(results, key=lambda x: x["tot"], reverse=True)
    return ranked[:top_k], ranked[0]


# ======================== MAIN ======================================= #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--top-k", type=int, default=20)

    # ADD EARLY-STOP CLI SUPPORT HERE
    parser.add_argument("--early-stop", action="store_true",
                        help="Enable early stopping for bad strategies.")
    parser.add_argument("--early-stop-trades", type=int, default=250,
                        help="Minimum trades before early stopping kicks in.")
    parser.add_argument("--early-stop-threshold", type=float, default=-1.0,
                        help="Stop if cumulative return <= this value.")

    args = parser.parse_args()

    df = load_data(args.csv)

    top_k_results, best = optimize(
        df,
        early_stop=args.early_stop,
        early_trades=args.early_stop_trades,
        early_threshold=args.early_stop_threshold,
        top_k=args.top_k
    )

    print("\n==================== TOP STRATEGIES ====================\n")
    for i, r in enumerate(top_k_results):
        print(f"#{i+1}  {r}")

    with open("best_strategy.json", "w") as f:
        json.dump(best, f, indent=4)

    print("\nBest strategy saved to best_strategy.json\n")


if __name__ == "__main__":
    main()
