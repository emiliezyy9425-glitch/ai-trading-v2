import argparse
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from tsla_ai_master_final_ready_backtest import get_historical_data_backtest

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/app")
TRADE_LOG_PATH = os.getenv("TRADE_LOG_PATH", os.path.join(PROJECT_ROOT, "data", "trade_log.csv"))

# Load environment variables
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

# Create logs directory before configuring logging
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)

# Setup logging to both generate_trade_log.log and retrain.log for compatibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, "logs", "generate_trade_log.log")),
        logging.FileHandler(os.path.join(PROJECT_ROOT, "logs", "retrain.log")),  # Align with launch_all.sh
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_fibonacci_levels(price, min_price, max_price):
    """Generate Fibonacci levels as a string based on price range."""
    try:
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        price_range = max_price - min_price
        if price_range == 0:
            logger.warning("Max price equals min price; returning empty fib levels.")
            return ""
        levels = [min_price + level * price_range for level in fib_levels]
        return ",".join([f"{fib_levels[i]}:{levels[i]:.2f}" for i in range(len(fib_levels))])
    except Exception as e:
        logger.error(f"Failed to generate Fibonacci levels: {e}")
        return ""

def calculate_rsi_series(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI as a series for each row, guarding against divide-by-zero and clipping to [0,100]."""
    try:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        # Safe RSI: handle zero loss/gain explicitly
        rsi = pd.Series(
            np.where(
                loss == 0, 100.0,
                np.where(
                    gain == 0, 0.0,
                    100 - (100 / (1 + (gain / loss)))
                )
            ),
            index=prices.index
        )
        # Clip to [0,100] for any anomalies
        rsi = rsi.clip(0, 100)
        return rsi
    except Exception as e:
        logger.error(f"Failed to calculate RSI: {e}")
        return pd.Series(np.nan, index=prices.index)

def calculate_macd_series(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    """Compute MACD and signal as series for each row."""
    try:
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    except Exception as e:
        logger.error(f"Failed to calculate MACD: {e}")
        return pd.Series(np.nan, index=prices.index), pd.Series(np.nan, index=prices.index)

def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def generate_trade_log(ticker: str, start: str, end: str, out: str):
    try:
        start_dt = _parse_date(start)
        end_dt = _parse_date(end)
        if start_dt >= end_dt:
            raise ValueError("start date must be earlier than end date")

        logger.info(
            "Loading %s 1-day bars from local backtest cache between %s and %s...",
            ticker,
            start_dt.date(),
            end_dt.date(),
        )

        df = get_historical_data_backtest(ticker, "1 day", start_dt, end_dt)
        if df is None or df.empty:
            logger.error(
                "No local historical data available for %s between %s and %s.",
                ticker,
                start,
                end,
            )
            raise ValueError("Missing local historical data")

        df = df.reset_index(drop=True)
        logger.info("Retrieved %d rows of %s data.", len(df), ticker)
        if len(df) < 50:
            logger.warning(
                "Only %d rows retrieved; consider expanding the requested window for better coverage.",
                len(df),
            )

        trade_log = pd.DataFrame()
        trade_log['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.strftime('%Y-%m-%d %H:%M:%S')
        trade_log['ticker'] = ticker
        trade_log['price'] = df['close'].round(2)
        trade_log['volume'] = df['volume'].astype(int)
        trade_log['RSI'] = calculate_rsi_series(df['close'])
        macd, signal = calculate_macd_series(df['close'])
        trade_log['MACD'] = macd
        trade_log['Signal'] = signal

        # Handle NaNs and round
        trade_log[["RSI", "MACD", "Signal"]] = trade_log[["RSI", "MACD", "Signal"]].bfill().astype(float).round(2)

        np.random.seed(42)
        trade_log['decision'] = np.random.choice(['Buy', 'Sell', 'Hold'], size=len(df))
        trade_log['Source'] = 'AI'
        min_price = df['low'].min()
        max_price = df['high'].max()
        trade_log['fib'] = [generate_fibonacci_levels(price, min_price, max_price) for price in trade_log['price']]
        trade_log['tds'] = 0
        trade_log['td9'] = 0
        trade_log['IV'] = np.random.uniform(0.3, 0.5, size=len(df)).round(2)
        trade_log['Delta'] = np.random.uniform(-1, 1, size=len(df)).round(2)

        trade_log = trade_log[[
            'timestamp', 'ticker', 'price', 'decision', 'fib', 'tds', 'td9',
            'RSI', 'MACD', 'Signal', 'volume', 'IV', 'Delta', 'Source'
        ]]

        # Ensure output directory exists
        os.makedirs(os.path.dirname(out), exist_ok=True)
        trade_log.to_csv(out, index=False)
        logger.info(f"trade_log.csv saved to {out} with {len(trade_log)} rows.")
        return True
    except Exception as e:
        logger.error(f"Failed to generate trade_log.csv: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate trade log CSV from locally cached historical data."
    )
    parser.add_argument("--ticker", type=str, default="TSLA", help="Ticker symbol (default: TSLA)")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date (YYYY-MM-DD, default: 2024-01-01)")
    parser.add_argument("--end", type=str, default=datetime.now(timezone.utc).strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD, default: today UTC)")
    parser.add_argument(
        "--out",
        type=str,
        default=TRADE_LOG_PATH,
        help="Output CSV path (default: TRADE_LOG_PATH or <PROJECT_ROOT>/data/trade_log.csv)",
    )

    args = parser.parse_args()
    try:
        success = generate_trade_log(args.ticker, args.start, args.end, args.out)
        if not success:
            sys.exit(1)
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        sys.exit(1)
