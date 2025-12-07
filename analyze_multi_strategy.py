#!/usr/bin/env python3
"""
Analyze whether using multiple strategies at the same time
can achieve higher total profit than any single strategy.

A "strategy" here is defined by:
- model name (rf, xgb, lgb, lstm, tcn, ppo, transformer)
- confidence threshold
- exit style (A, B, C, D, E)

We:
1. Backtest all combinations in a grid.
2. Rank them by total return (sum of % returns).
3. Build a "multi-strategy portfolio" by combining several strategies.

ASSUMPTION for multi-strategy combo:
- Each strategy uses its **own capital**, i.e. you can run them in parallel.
- So combined total return = sum of individual total returns.
- This answers: "If I run multiple strategies at once, which mix maximizes profit?"
"""

import argparse
from itertools import product
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm


# ------------------------- Configurable Grid -------------------------- #

# Models to consider
MODELS = ["rf", "xgb", "lgb", "lstm", "tcn", "ppo", "transformer"]

# Exit styles:
# A: fixed 4h exit
# B: fixed 8h exit
# C: TP/SL only (with max hold)
# D: reverse-signal only
# E: hybrid (TP/SL + reverse + max hold)
EXIT_STYLES = ["A", "B", "C", "D", "E"]

# Confidence thresholds to test, from 1.0 down to 0.5 (step 0.01)
CONF_THRESHOLDS = [round(th, 2) for th in np.linspace(1.0, 0.5, 51)]

# TP / SL used for exit styles C and E
TP_DEFAULT = 0.02   # +2%
SL_DEFAULT = 0.01   # -1%
MAX_HOURS_DEFAULT = 24.0


# ---------------------------- Backtester ------------------------------ #

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


def backtest_model_single(
    df: pd.DataFrame,
    model: str,
    conf_thresh: float,
    exit_style: str,
    tp: float = TP_DEFAULT,
    sl: float = SL_DEFAULT,
    max_hours: float = MAX_HOURS_DEFAULT,
    h4: float = 4.0,
    h8: float = 8.0,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Backtest a single model with given confidence threshold and exit style.

    Returns:
        summary dict:
            {n_trades, total_return, win_rate, model, conf_thresh, exit_style}
        trades DataFrame (entry/exit for each trade)
    """
    vote_col = f"{model}_vote"
    conf_col = f"{model}_conf"

    if vote_col not in df.columns or conf_col not in df.columns:
        # No such model in the data
        return {
            "n": 0,
            "tot": 0.0,
            "avg": 0.0,
            "winrate": 0.0,
            "model": model,
            "conf_thresh": conf_thresh,
            "exit_style": exit_style,
        }, pd.DataFrame()

    trades = []

    # Run per-ticker to avoid mixing instruments
    for ticker, g in df.groupby("ticker"):
        g = g.reset_index(drop=True)
        pos = None  # {t0, p0, dir}

        for _, row in g.iterrows():
            t = row["timestamp"]
            price = row["price"]
            vote = row[vote_col]
            conf = row[conf_col]

            # manage exit if a position is open
            if pos is not None:
                dt_hours = (t - pos["t0"]).total_seconds() / 3600.0
                ret = (price - pos["p0"]) / pos["p0"]
                if pos["dir"] == "SELL":
                    ret = -ret  # SELL profits when price goes down

                exit_now = False
                reason = None

                if exit_style in ("A", "B"):
                    H = h4 if exit_style == "A" else h8
                    if dt_hours >= H:
                        exit_now = True
                        reason = f"time_{H}h"

                elif exit_style in ("C", "E"):
                    # TP/SL logic
                    if ret >= tp:
                        exit_now = True
                        reason = "tp"
                    elif ret <= -sl:
                        exit_now = True
                        reason = "sl"

                    # Max holding time
                    if not exit_now and dt_hours >= max_hours:
                        exit_now = True
                        reason = f"max_{max_hours}h"

                if exit_style in ("D", "E") and not exit_now:
                    # reverse-signal exit: close when model flips direction
                    if vote in ("BUY", "SELL") and vote != pos["dir"]:
                        exit_now = True
                        reason = "reverse"

                if exit_now:
                    trades.append(
                        {
                            "ticker": ticker,
                            "model": model,
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

            # manage entry if flat
            if pos is None:
                if vote in ("BUY", "SELL") and conf >= conf_thresh:
                    pos = {"t0": t, "p0": price, "dir": vote}

        # At end of ticker history, close any remaining position at last bar
        if pos is not None:
            price = g["price"].iloc[-1]
            t_last = g["timestamp"].iloc[-1]
            ret = (price - pos["p0"]) / pos["p0"]
            if pos["dir"] == "SELL":
                ret = -ret
            trades.append(
                {
                    "ticker": ticker,
                    "model": model,
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
            "n": 0,
            "tot": 0.0,
            "avg": 0.0,
            "winrate": 0.0,
            "model": model,
            "conf_thresh": conf_thresh,
            "exit_style": exit_style,
        }
        return summary, tr_df

    n_trades = len(tr_df)
    total_return = tr_df["ret"].sum()       # sum of % returns
    avg_return = tr_df["ret"].mean()        # avg % per trade
    winrate = (tr_df["ret"] > 0).mean()     # fraction of winning trades

    summary = {
        "n": n_trades,
        "tot": float(total_return),
        "avg": float(avg_return),
        "winrate": float(winrate),
        "model": model,
        "conf_thresh": conf_thresh,
        "exit_style": exit_style,
    }
    return summary, tr_df


# ------------------------ Multi-Strategy Logic ------------------------ #

def build_multi_strategy_portfolio(
    summaries: pd.DataFrame,
    strategy_filter: str = "positive",  # or "top_k"
    top_k: int = 10,
) -> Dict:
    """
    Combine multiple strategies to see if running them together
    increases total profit.

    We use SUMMARY ONLY (no alignment in time):

      total_return_portfolio = sum(tot_i)
      n_trades_portfolio     = sum(n_i)
      avg_return_portfolio   = total_return_portfolio / n_trades_portfolio
      winrate_portfolio      = weighted by number of trades

    strategy_filter:
      "positive" : use all strategies with tot > 0
      "top_k"    : use top_k strategies by tot
    """
    df = summaries.copy()

    if strategy_filter == "positive":
        combo_df = df[df["tot"] > 0].copy()
    elif strategy_filter == "top_k":
        combo_df = df.sort_values("tot", ascending=False).head(top_k).copy()
    else:
        raise ValueError("strategy_filter must be 'positive' or 'top_k'")

    if combo_df.empty:
        return {
            "n_strategies": 0,
            "n_trades": 0,
            "total_return": 0.0,
            "avg_return": 0.0,
            "winrate": 0.0,
        }

    total_return = combo_df["tot"].sum()
    total_trades = combo_df["n"].sum()

    if total_trades > 0:
        avg_return = total_return / total_trades
        winrate = (combo_df["winrate"] * combo_df["n"]).sum() / total_trades
    else:
        avg_return = 0.0
        winrate = 0.0

    return {
        "n_strategies": int(len(combo_df)),
        "n_trades": int(total_trades),
        "total_return": float(total_return),
        "avg_return": float(avg_return),
        "winrate": float(winrate),
    }


# ------------------------------ Main ---------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Search single and multi-strategy combinations for maximum profit."
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
        help="Number of best single strategies to print (default: 20)",
    )
    args = parser.parse_args()

    df = load_data(args.csv)

    summaries: List[Dict] = []
    # If you want trade-level data for later inspection, you can store them too.

    # 1) Backtest all single strategies in the grid
    total_runs = len(MODELS) * len(CONF_THRESHOLDS) * len(EXIT_STYLES)
    for model, conf, style in tqdm(
        product(MODELS, CONF_THRESHOLDS, EXIT_STYLES),
        total=total_runs,
        desc="Backtesting strategies",
        unit="strategy",
    ):
        summary, _ = backtest_model_single(
            df,
            model=model,
            conf_thresh=conf,
            exit_style=style,
            tp=TP_DEFAULT,
            sl=SL_DEFAULT,
            max_hours=MAX_HOURS_DEFAULT,
        )
        summaries.append(summary)

    res_df = pd.DataFrame(summaries)

    # Sort single strategies by total return
    res_df = res_df.sort_values("tot", ascending=False).reset_index(drop=True)

    print("\n==== Top Single Strategies ====\n")
    print(res_df.head(args.top_k).to_string(index=False))

    # 2) Multi-strategy: use all positive strategies
    combo_pos = build_multi_strategy_portfolio(res_df, strategy_filter="positive")
    print("\n==== Multi-Strategy Portfolio: All Positive Strategies ====\n")
    print(combo_pos)

    # 3) Multi-strategy: use top-K single strategies
    combo_topk = build_multi_strategy_portfolio(
        res_df, strategy_filter="top_k", top_k=args.top_k
    )
    print(f"\n==== Multi-Strategy Portfolio: Top {args.top_k} Strategies ====\n")
    print(combo_topk)


if __name__ == "__main__":
    main()
