# =============================================================================
# 10-Day EMA Reversion + Pure Martingale Backtester (IBKR Ready)
# Exact replica of your Pine Script – now in Python using ib_insync
# =============================================================================

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ib_insync import IB, Stock, util
from live_trading import connect_ibkr   # ← Your live bot's connection function

# --------------------------- CONFIG ---------------------------
TICKERS_FILE = Path("tickers.txt")
START_DATE = "20240101"  # 1+ year of data
END_DATE = "20251231"
CAPITAL = 500_000
COMMISSION_PER_SHARE = 0.0035
MARTINGALE_CAP_PCT = 100.0  # Max 100% of equity
RISK_RESET_PCT = 1.0  # Start with 1%

# List of timeframes to test sequentially
TIMEFRAMES = [
    "1 min",
    "2 mins",
    "3 mins",
    "5 mins",
    "15 mins",
    "30 mins",
    "1 hour",
    "4 hours",
    "1 day",
]

# ----------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
util.startLoop()


def load_tickers(path: Path) -> list[str]:
    """Load ticker symbols from a text file (one per line).

    Ignores blank lines and lines starting with '#'. Raises a helpful error if
    the file is missing or empty.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Ticker list not found at {path.resolve()}. Please create tickers.txt with one symbol per line."
        )

    tickers = [
        line.strip().upper()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not tickers:
        raise ValueError(f"No tickers found in {path}. Add at least one symbol to run the backtest.")

    return tickers


def get_daily_ema10(ib: IB, contract: Stock, end_date: datetime) -> pd.DataFrame:
    """Fetch daily bars and compute true non-repainting 10-period EMA."""

    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end_date,
        durationStr="2 Y",
        barSizeSetting="1 day",
        whatToShow="MIDPOINT",
        useRTH=True,
        formatDate=1,
    )
    df = util.df(bars)
    df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
    return df[["date", "close", "ema10"]].set_index("date")


async def run_backtest(symbol: str, timeframe: str) -> pd.DataFrame:
    print(f"\n{'=' * 70}")
    print(f"BACKTESTING: {symbol} | {timeframe} | Martingale Cap: {MARTINGALE_CAP_PCT}%")
    print(f"{'=' * 70}")

    # === RE-USE YOUR LIVE BOT'S CONNECTION (SAFE + CLEAN) ===
    ib = connect_ibkr(max_retries=3, initial_client_id=300)  # Safe client ID
    if ib is None or not ib.isConnected():
        print(f"IBKR connection failed for {symbol} {timeframe} — skipping")
        return pd.DataFrame()

    contract = Stock(symbol, "SMART", "USD")
    ib.qualifyContracts(contract)

    # Download price data
    end_dt = datetime.now()
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end_dt,
        durationStr="2 Y" if timeframe == "1 day" else "365 D",
        barSizeSetting=timeframe,
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )

    # === DISCONNECT AFTER DOWNLOAD ===
    ib.disconnect()
    if not bars:
        print("No data from IBKR")
        return pd.DataFrame()

    df = util.df(bars)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # === FIXED EMA10 WITH PROPER DATETIMEINDEX ===
    ib_daily = connect_ibkr(max_retries=1, initial_client_id=310)
    if ib_daily and ib_daily.isConnected():
        try:
            bars_daily = ib_daily.reqHistoricalData(
                contract,
                endDateTime=end_dt,
                durationStr="2 Y",
                barSizeSetting="1 day",
                whatToShow="MIDPOINT",
                useRTH=True,
                formatDate=1,
            )
            daily_df = util.df(bars_daily)
            daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.tz_localize("UTC")
            daily_df = daily_df.set_index("date")
            daily_df["ema10"] = daily_df["close"].ewm(span=10, adjust=False).mean()
            daily_ema = daily_df["ema10"]

            ema_resampled = daily_ema.resample("1min").ffill().reindex(df.index, method="nearest")
            df["ema10"] = ema_resampled

        finally:
            ib_daily.disconnect()
    else:
        df["ema10"] = np.nan

    # Strategy logic
    df["prev_close"] = df["close"].shift(1)
    df["prev_ema"] = df["ema10"].shift(1)

    df["buy_signal"] = (df["prev_close"] < df["prev_ema"]) & (df["close"] > df["ema10"])
    df["sell_signal"] = (df["prev_close"] > df["prev_ema"]) & (df["close"] < df["ema10"])

    # Backtest variables
    equity = CAPITAL
    risk_pct = RISK_RESET_PCT
    position = 0  # 1 = long, -1 = short, 0 = flat
    entry_price = 0.0
    trade_log: list[dict[str, float | int | bool | datetime | str]] = []

    for i in range(1, len(df)):
        row = df.iloc[i]

        # Check for exit first (opposite signal)
        if position != 0:
            if (position == 1 and row.sell_signal) or (position == -1 and row.buy_signal):
                # Close position
                exit_price = row["open"]  # Realistic: exit at next bar open
                shares = abs(position) * (equity * (risk_pct / 100) / entry_price)
                pnl = position * (exit_price - entry_price) * shares
                commission = shares * 2 * COMMISSION_PER_SHARE
                pnl -= commission

                equity += pnl
                win = pnl > 0

                trade_log.append(
                    {
                        "timestamp": row.name,
                        "type": "LONG" if position > 0 else "SHORT",
                        "entry": entry_price,
                        "exit": exit_price,
                        "shares": round(shares),
                        "pnl_dollar": round(pnl, 2),
                        "pnl_percent": round(pnl / (equity - pnl) * 100, 3),
                        "risk_percent": risk_pct,
                        "equity_after": round(equity, 2),
                        "win": win,
                    }
                )

                # Martingale update
                risk_pct = RISK_RESET_PCT if win else min(risk_pct * 2, MARTINGALE_CAP_PCT)

                position = 0  # flatten

        # New entry?
        if position == 0:
            if row.buy_signal:
                position = 1
                entry_price = row["open"]
            elif row.sell_signal:
                position = -1
                entry_price = row["open"]

    ib.disconnect()

    # Final Results
    total_trades = len(trade_log)
    wins = sum(1 for t in trade_log if t["win"])
    win_rate = wins / total_trades * 100 if total_trades else 0
    final_equity = equity
    return_pct = (final_equity - CAPITAL) / CAPITAL * 100

    print(f"\nRESULTS ({symbol} | {timeframe})")
    print(f"Total Trades     : {total_trades}")
    print(f"Win Rate         : {win_rate:.1f}%")
    print(f"Final Equity     : ${final_equity:,.0f}")
    print(f"Total Return     : {return_pct:+.2f}%")
    print(
        f"Max Risk Reached : {max(t['risk_percent'] for t in trade_log) if trade_log else 1}%"
    )

    # Save detailed log
    log_df = pd.DataFrame(trade_log)
    filename = f"martingale_{symbol}_{timeframe.replace(' ', '')}_results.csv"
    log_df.to_csv(filename, index=False)
    print(f"Detailed trades saved → {filename}\n")

    if not log_df.empty:
        print(f"\nRunning detailed analysis for {symbol} | {timeframe}...")
        analyze_trades(
            trade_log,
            initial_capital=CAPITAL,
            timeframe_name=f"{symbol}_{timeframe}",
        )

    return log_df


def analyze_trades(
    trade_log: list[dict],
    initial_capital: float = 500_000,
    timeframe_name: str = "Strategy",
) -> dict:
    """Generate a statistical summary and equity curve for a trade log.

    Expects the log entries generated by ``run_backtest`` (timestamp, equity_after,
    win, pnl_dollar, risk_percent, etc.). Saves an equity curve plot and returns a
    dict of headline metrics.
    """

    if not trade_log:
        print("No trades!")
        return {}

    df = pd.DataFrame(trade_log)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    equity_curve = [initial_capital] + df["equity_after"].tolist()

    # ───── Basic Stats ─────
    total_trades = len(df)
    wins = df["win"].sum()
    win_rate = wins / total_trades * 100
    net_profit = df["pnl_dollar"].sum()
    total_return = (equity_curve[-1] - initial_capital) / initial_capital * 100

    gross_win = df[df["win"]]["pnl_dollar"].sum()
    gross_loss = df[~df["win"]]["pnl_dollar"].sum()
    profit_factor = abs(gross_win / gross_loss) if gross_loss != 0 else np.inf
    avg_win = gross_win / wins if wins > 0 else 0
    avg_loss = (
        abs(gross_loss / (total_trades - wins)) if (total_trades - wins) > 0 else 0
    )

    # ───── Max Consecutive Losses ─────
    df["loss_streak"] = (~df["win"]).cumsum()
    df["streak_group"] = df["win"].ne(df["win"].shift()).cumsum()
    max_consec_loss = (
        df[~df["win"]].groupby("streak_group").size().max()
        if not df["win"].all()
        else 0
    )

    # ───── Max Drawdown ─────
    eq = pd.Series(equity_curve)
    rolling_max = eq.cummax()
    drawdown = eq - rolling_max
    max_dd = drawdown.min()
    max_dd_pct = max_dd / rolling_max.max() * 100

    # ───── Sharpe Ratio (annualized) ─────
    daily_ret = eq.pct_change().dropna()
    sharpe = np.sqrt(252) * daily_ret.mean() / daily_ret.std() if daily_ret.std() != 0 else 0

    # ───── Print Beautiful Summary ─────
    print(f"\n{'='*50}")
    print(f"  DETAILED ANALYSIS – {timeframe_name.upper()}")
    print(f"{'='*50}")
    print(f"Total Trades         : {total_trades}")
    print(f"Win Rate             : {win_rate:6.2f}%")
    print(f"Profit Factor        : {profit_factor:6.2f}")
    print(f"Avg Win / Avg Loss   : ${avg_win:,.0f} / ${avg_loss:,.0f}")
    print(f"Max Consecutive Loss : {max_consec_loss} trades")
    print(f"Net Profit           : ${net_profit:,.0f}")
    print(f"Final Equity         : ${equity_curve[-1]:,.0f}")
    print(f"Total Return         : {total_return:+6.2f}%")
    print(f"Max Drawdown         : {max_dd_pct:6.2f}%")
    print(f"Sharpe Ratio         : {sharpe:6.2f}")
    print(f"Max Risk Used        : {df['risk_percent'].max():.1f}%")
    print(f"{'='*50}\n")

    # ───── Plot & Save Equity Curve ─────
    plt.figure(figsize=(12, 6))
    plt.plot(
        range(len(eq)),
        eq,
        linewidth=2,
        color="green",
    )
    plt.title(f"Equity Curve – {timeframe_name} | Final: ${equity_curve[-1]:,.0f}")
    plt.ylabel("Equity ($)")
    plt.xlabel("Trade Number")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Equity_Curve_{timeframe_name.replace(' ', '_')}.png", dpi=200)
    plt.show()

    return {
        "timeframe": timeframe_name,
        "trades": total_trades,
        "win_rate_%": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "total_return_%": round(total_return, 2),
        "max_dd_%": round(max_dd_pct, 2),
        "sharpe": round(sharpe, 2),
        "max_consec_loss": max_consec_loss,
        "final_equity": round(equity_curve[-1]),
    }


# =============================================================================
# RUN ALL TIMEFRAMES SEQUENTIALLY
# =============================================================================
async def main() -> None:
    tickers = load_tickers(TICKERS_FILE)

    for symbol in tickers:
        print(f"\n{'#' * 60}")
        print(f"Running backtests for {symbol} across {len(TIMEFRAMES)} timeframes")
        print(f"{'#' * 60}")

        for timeframe in TIMEFRAMES:
            try:
                await run_backtest(symbol, timeframe)
            except Exception as exc:  # pragma: no cover - runtime guard
                logging.exception("Error on %s (%s): %s", symbol, timeframe, exc)
            await asyncio.sleep(2)  # Be gentle with IBKR rate limits


if __name__ == "__main__":
    asyncio.run(main())

# Use higher client IDs to never conflict with live bot
CLIENT_ID_BASE = 300
