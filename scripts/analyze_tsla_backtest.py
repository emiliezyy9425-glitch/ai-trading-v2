"""
Analyze TSLA backtest trade logs with configurable exit logic and confidence thresholds.

Example:
    python scripts/analyze_tsla_backtest.py \
        --csv trade_log_backtest_all_tickers.csv \
        --ticker TSLA \
        --top 20
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/app"))
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_BACKTEST_LOG = DATA_DIR / "trade_log_backtest_all_tickers.csv"

# Models to evaluate; extend this list to include new model families.
MODELS: List[str] = ["rf", "xgb", "lgb", "lstm", "tcn", "ppo", "transformer"]
EXIT_STYLES: List[str] = ["A", "B", "C", "D", "E"]
DEFAULT_THRESHOLD_GRID: List[float] = [0.90, 0.95, 0.97, 0.98, 0.99]


def prepare_dataframe(csv_path: Path, ticker: str | None = "TSLA") -> pd.DataFrame:
    """Load and normalize the trade log.

    Votes are uppercased and confidences coerced to numeric. If ``ticker`` is
    provided, the dataset is filtered to that symbol.
    """

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    for c in [f"{m}_vote" for m in MODELS]:
        df[c] = df[c].astype(str).str.upper()
    for c in [f"{m}_conf" for m in MODELS]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if ticker:
        df = df[df["ticker"].str.upper() == ticker.upper()].reset_index(drop=True)
    return df


def backtest_model(
    df: pd.DataFrame,
    model: str,
    conf_thresh: float,
    exit_style: str,
    tp: float = 0.02,
    sl: float = 0.01,
    max_hours: int = 24,
    h4: int = 4,
    h8: int = 8,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Backtest a single model using the provided exit style and thresholds."""

    vote_col = f"{model}_vote"
    conf_col = f"{model}_conf"
    trades: List[Dict[str, object]] = []

    for ticker, group in df.groupby("ticker"):
        group = group.reset_index(drop=True)
        open_position = None
        for _, row in group.iterrows():
            timestamp = row["timestamp"]
            price = row["price"]
            vote = row[vote_col]
            conf = row[conf_col]

            if open_position is not None:
                delta_hours = (timestamp - open_position["t0"]).total_seconds() / 3600.0
                ret = (price - open_position["p0"]) / open_position["p0"]
                if open_position["dir"] == "SELL":
                    ret = -ret

                exit_now = False
                reason = None

                if exit_style in ["A", "B"]:
                    hold_hours = h4 if exit_style == "A" else h8
                    if delta_hours >= hold_hours:
                        exit_now = True
                        reason = f"time_{hold_hours}h"
                elif exit_style in ["C", "E"]:
                    if ret >= tp:
                        exit_now = True
                        reason = "tp"
                    elif ret <= -sl:
                        exit_now = True
                        reason = "sl"
                    if not exit_now and delta_hours >= max_hours:
                        exit_now = True
                        reason = f"max_{max_hours}h"

                if exit_style in ["D", "E"] and not exit_now:
                    if vote in ["BUY", "SELL"] and vote != open_position["dir"]:
                        exit_now = True
                        reason = "reverse"

                if exit_now:
                    trades.append(
                        {
                            "model": model,
                            "ticker": ticker,
                            "entry_time": open_position["t0"],
                            "exit_time": timestamp,
                            "direction": open_position["dir"],
                            "entry_price": open_position["p0"],
                            "exit_price": price,
                            "ret": ret,
                            "reason": reason,
                        }
                    )
                    open_position = None

            if open_position is None and vote in ["BUY", "SELL"] and conf >= conf_thresh:
                open_position = {"t0": timestamp, "p0": price, "dir": vote}

        if open_position is not None:
            price = group["price"].iloc[-1]
            timestamp = group["timestamp"].iloc[-1]
            ret = (price - open_position["p0"]) / open_position["p0"]
            if open_position["dir"] == "SELL":
                ret = -ret
            trades.append(
                {
                    "model": model,
                    "ticker": ticker,
                    "entry_time": open_position["t0"],
                    "exit_time": timestamp,
                    "direction": open_position["dir"],
                    "entry_price": open_position["p0"],
                    "exit_price": price,
                    "ret": ret,
                    "reason": "eod",
                }
            )

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return {"n": 0, "tot": 0.0, "avg": 0.0, "winrate": 0.0}, trades_df

    return {
        "n": len(trades_df),
        "tot": float(trades_df["ret"].sum()),
        "avg": float(trades_df["ret"].mean()),
        "winrate": float((trades_df["ret"] > 0).mean()),
    }, trades_df


def run_grid(
    df: pd.DataFrame,
    models: Iterable[str],
    exit_styles: Iterable[str],
    threshold_grid: Iterable[float],
) -> pd.DataFrame:
    """Evaluate all model/exit style/confidence combinations."""

    results: List[Dict[str, object]] = []
    for model in models:
        for style in exit_styles:
            for threshold in threshold_grid:
                summary, _ = backtest_model(df, model, threshold, style)
                summary.update({"model": model, "exit_style": style, "conf_thresh": threshold})
                results.append(summary)
    return pd.DataFrame(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_BACKTEST_LOG,
        help=f"Path to the trade log CSV file (default: {DEFAULT_BACKTEST_LOG}).",
    )
    parser.add_argument("--ticker", type=str, default="TSLA", help="Ticker to filter by (default: TSLA). Use blank to keep all.")
    parser.add_argument("--top", type=int, default=20, help="Number of top rows to display by total return.")
    parser.add_argument("--save-results", type=Path, default=None, help="Optional path to save the aggregated results CSV.")
    parser.add_argument(
        "--save-trades", type=Path, default=None, help="Optional path to save the detailed trades CSV for the best combo."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = prepare_dataframe(args.csv, args.ticker or None)

    res_df = run_grid(df, MODELS, EXIT_STYLES, DEFAULT_THRESHOLD_GRID)
    res_df = res_df.sort_values("tot", ascending=False).reset_index(drop=True)
    print(res_df.head(args.top))

    if args.save_results is not None:
        res_df.to_csv(args.save_results, index=False)
        print(f"Saved aggregated results to {args.save_results}")

    if args.save_trades is not None:
        best = res_df.iloc[0]
        summary, trades = backtest_model(
            df,
            best["model"],
            float(best["conf_thresh"]),
            best["exit_style"],
        )
        trades.to_csv(args.save_trades, index=False)
        print(f"Saved trades for best combo ({summary}) to {args.save_trades}")


if __name__ == "__main__":
    main()
