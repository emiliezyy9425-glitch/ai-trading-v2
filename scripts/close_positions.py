import os
import logging
from typing import List

try:  # pragma: no cover - import resolution differs between execution modes
    from ._bootstrap import ensure_project_root_on_path  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - fallback when running as a script
    from _bootstrap import ensure_project_root_on_path  # type: ignore[import-not-found]

ensure_project_root_on_path()


from tsla_ai_master_final_ready import (
    connect_ibkr,
    close_option_positions,
    close_stock_position,
    get_stock_price,
    get_position_info,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _read_tickers(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        logger.error(f"Tickers file not found: {file_path}")
        return []
    with open(file_path) as f:
        tickers = [t.strip() for t in f if t.strip()]
    return [t for t in tickers if t.upper() != "TSLA"]


def close_positions(file_path: str = os.path.join("data", "tickers.txt")) -> None:
    tickers = _read_tickers(file_path)
    if not tickers:
        logger.info("No tickers to close.")
        return

    ib = connect_ibkr()
    if ib is None or not ib.isConnected():
        logger.error("Failed to connect to IBKR. Aborting.")
        return

    for ticker in tickers:
        try:
            option_closed = close_option_positions(ib, ticker, require_profit=True)
            pos_qty, avg_cost = get_position_info(ib, ticker)
            if pos_qty <= 0:
                # No equity holdings remain; flatten any outstanding options to avoid naked exposure.
                all_options_closed = close_option_positions(ib, ticker)
                if all_options_closed:
                    logger.info(
                        f"Closed option positions for {ticker}; no stock holdings to close."
                    )
                elif option_closed:
                    logger.info(
                        f"Closed profitable option positions for {ticker}; no stock holdings to close."
                    )
                else:
                    logger.info(f"No open stock position for {ticker}; skipping")
                continue

            price = get_stock_price(ib, ticker)
            if price is None:
                logger.warning(f"Skipping {ticker}: unable to fetch price")
                continue

            if price <= avg_cost:
                logger.info(
                    f"Skipping {ticker}: not in profit (price {price:.2f} <= avg cost {avg_cost:.2f})"
                )
                continue

            close_stock_position(ib, ticker, price, allow_after_hours=True)
            close_option_positions(ib, ticker)
        except Exception as e:
            logger.error(f"Failed to close position for {ticker}: {e}")

    ib.disconnect()


if __name__ == "__main__":
    close_positions()
