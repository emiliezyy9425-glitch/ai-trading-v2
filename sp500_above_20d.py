import logging
import re
import pandas as pd
import httpx
from ib_insync import *
from typing import Optional
from pathlib import Path
from indicators import get_historical_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/sp500_above_20d.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_sp500_above_20d_investing():
    """Fetch the official breadth indicator (S5TW) from Investing.com.

    Returns:
        tuple: (pct_above_ma, None, None) or neutral defaults on failure.
    """
    url = "https://www.investing.com/indices/s-p-500-stocks-above-20-day-average"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = httpx.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        match = re.search(r'data-test="instrument-price-last">([0-9,.]+)', resp.text)
        if match:
            pct_above_ma = float(match.group(1).replace(",", ""))
        else:
            tables = pd.read_html(resp.text)
            pct_above_ma = float(tables[0].iloc[0, 1])
        logger.info(
            f"✅ Investing.com S5TW fetched successfully: {pct_above_ma:.2f}% above 20-day MA"
        )
        return pct_above_ma, None, None
    except Exception as e:  # pragma: no cover - network issues
        logger.error(f"❌ Error fetching S5TW from Investing.com: {e}")
        return 50.0, None, None


def get_sp500_above_20d(ib: Optional[IB] = None):
    """Fetch S&P 500 breadth via IBKR with Investing.com fallback.

    Args:
        ib: IBKR connection or ``None``.

    Returns:
        tuple: (pct_above_ma, spx_close, ma20)
    """
    try:
        if ib is None or not ib.isConnected():
            logger.warning(
                "⚠️ No IBKR connection. Fetching S5TW from Investing.com instead."
            )
            return get_sp500_above_20d_investing()

        # Fetch SPX data
        contract = Index("SPX", "CBOE", "USD")
        ib.qualifyContracts(contract)
        df = get_historical_data(ib, "SPX", timeframe="1 day", duration="21 D")
        if df is None or df.empty:
            logger.error("❌ Failed to fetch SPX data from IBKR. Falling back to Investing.com.")
            return get_sp500_above_20d_investing()

        # Calculate 20-day MA and percentage above
        ma20 = df["close"].rolling(window=20).mean().iloc[-1]
        spx_close = df["close"].iloc[-1]
        above_ma = df["close"] > df["close"].rolling(window=20).mean()
        pct_above_ma = (
            (above_ma.sum() / len(above_ma)) * 100 if len(above_ma) > 0 else 50.0
        )

        logger.info(
            f"✅ S&P 500: {pct_above_ma:.2f}% above 20-day MA, Close: {spx_close:.2f}, MA20: {ma20:.2f}"
        )
        return pct_above_ma, spx_close, ma20
    except Exception as e:  # pragma: no cover - IBKR failure
        logger.error(f"❌ Error fetching S&P 500 data: {e}. Falling back to Investing.com.")
        return get_sp500_above_20d_investing()


def load_sp500_above_20d_history(csv_path: Path | str | None = None) -> pd.Series:
    """Load historical S&P 500 breadth from a CSV file.

    The CSV is expected to contain ``Date`` and ``Price`` columns as exported
    from Investing.com. Dates are parsed in ``MM/DD/YYYY`` format and the
    returned series is indexed by ``datetime.date`` for easy joining with other
    datasets.

    Args:
        csv_path: Optional path to the CSV file. When ``None``, the function
            looks for ``data/S&P 500 Stocks Above 20-Day Average Historical Data.csv``
            relative to the project root.

    Returns:
        pandas.Series: Series indexed by date with percentage values. An empty
        series is returned if parsing fails.
    """

    if csv_path is None:
        csv_path = Path(__file__).resolve().parent / "data" / (
            "S&P 500 Stocks Above 20-Day Average Historical Data.csv"
        )
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig", usecols=["Date", "Price"])
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        df["Price"] = pd.to_numeric(
            df["Price"].astype(str).str.replace(",", ""), errors="coerce"
        )
        df = df.dropna(subset=["Date", "Price"]).sort_values("Date")
        return df.set_index("Date")["Price"].rename("sp500_above_20d")
    except Exception as e:  # pragma: no cover - file issues
        logger.error(f"❌ Error loading S&P 500 breadth history: {e}")
        return pd.Series(dtype=float)


def get_sp500_above_20d_history(csv_path: Path | str | None = None) -> pd.Series:
    """Convenience wrapper to load historical S&P 500 breadth.

    This simply proxies to :func:`load_sp500_above_20d_history` with the
    default CSV bundled in ``data/S&P 500 Stocks Above 20-Day Average Historical Data.csv``.

    Args:
        csv_path: Optional custom path to the CSV file.

    Returns:
        pandas.Series: Series of percentage values indexed by ``datetime.date``.
    """

    return load_sp500_above_20d_history(csv_path)
