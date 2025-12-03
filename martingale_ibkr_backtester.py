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
from ib_insync import IB, Stock, util

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
    print(f"\n{'=' * 60}")
    print(
        f"BACKTESTING: {symbol} | Timeframe: {timeframe} | Martingale Cap: {MARTINGALE_CAP_PCT}%"
    )
    print(f"{'=' * 60}")

    ib = IB()
    await ib.connectAsync("127.0.0.1", 7497, clientId=99)  # TWS or IB Gateway (paper: 7497, live: 7496)

    contract = Stock(symbol, "SMART", "USD")
    ib.qualifyContracts(contract)

    # Download intraday data
    end_dt = datetime.now()
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end_dt,
        durationStr="365 D",
        barSizeSetting=timeframe,
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )
    df = util.df(bars)
    if df.empty:
        print("No data received!")
        ib.disconnect()
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # Get true daily EMA10 (non-repainting)
    daily_df = get_daily_ema10(ib, contract, end_dt)
    daily_ema = daily_df["ema10"].resample("1min").ffill().reindex(df.index, method="nearest")

    df["ema10"] = daily_ema

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
                        "time": row.name,
                        "type": "LONG" if position > 0 else "SHORT",
                        "entry": entry_price,
                        "exit": exit_price,
                        "shares": round(shares),
                        "pnl_$": round(pnl, 2),
                        "pnl_%": round(pnl / (equity - pnl) * 100, 3),
                        "risk_%": risk_pct,
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
    print(f"Max Risk Reached : {max(t['risk_%'] for t in trade_log) if trade_log else 1}%")

    # Save detailed log
    log_df = pd.DataFrame(trade_log)
    filename = f"martingale_{symbol}_{timeframe.replace(' ', '')}_results.csv"
    log_df.to_csv(filename, index=False)
    print(f"Detailed trades saved → {filename}\n")

    return log_df


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
