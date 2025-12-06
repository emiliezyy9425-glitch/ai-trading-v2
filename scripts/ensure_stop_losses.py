"""Ensure open positions have protective stop-loss orders in place.

This utility connects to Interactive Brokers, checks the configured tickers for
open stock positions, and submits a stop-loss order when none is active. Stop
prices are derived from the per-ticker configuration in
``tsla_ai_master_final_ready._model_stop_loss_pct`` using the specified model
name (defaults to ``LSTM``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List

# Ensure the repository root is on sys.path before importing project modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from ib_insync import Stock

from ibkr_utils import round_to_min_tick
from tickers_cache import load_tickers
from tsla_ai_master_final_ready import (
    connect_ibkr,
    get_position_info,
    _compute_stop_loss_price,
    _model_stop_loss_pct,
    _place_stop_loss_order,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_ACTIVE_STOP_STATUSES = {
    "PreSubmitted",
    "Submitted",
    "PendingSubmit",
    "PendingCancel",
}


def _iter_active_stop_orders(ib, ticker: str, action: str):
    """Yield active stop orders for *ticker* that match *action*.

    The helper inspects ``ib.openTrades()`` after refreshing open orders to avoid
    missing recently placed stops. Orders that are already filled or cancelled
    are filtered out.
    """

    try:
        ib.reqOpenOrders()
        ib.sleep(1)
    except Exception as exc:  # pragma: no cover - defensive guard for connectivity issues
        logger.warning("Unable to refresh open orders: %s", exc)

    try:
        trades = ib.openTrades()
    except Exception as exc:  # pragma: no cover - defensive guard for connectivity issues
        logger.error("Failed to fetch open trades: %s", exc)
        return

    for trade in trades:
        order = getattr(trade, "order", None)
        contract = getattr(trade, "contract", None)
        status = getattr(trade, "orderStatus", None)

        if not order or not contract:
            continue

        status_text = getattr(status, "status", "").title()
        if status_text and status_text not in _ACTIVE_STOP_STATUSES:
            continue

        if getattr(contract, "symbol", "") != ticker:
            continue
        if getattr(contract, "secType", "").upper() != "STK":
            continue

        order_type = getattr(order, "orderType", "").upper()
        order_action = getattr(order, "action", "").upper()
        if not order_type.startswith("STP"):
            continue
        if order_action != action.upper():
            continue

        yield trade


def _stop_loss_action(side: str) -> str:
    return "SELL" if side.upper() == "LONG" else "BUY"


def _ensure_stop_for_ticker(ib, ticker: str, model: str) -> None:
    position, avg_cost = get_position_info(ib, ticker)
    if position == 0:
        logger.info("No open position for %s; skipping stop-loss check.", ticker)
        return

    side = "LONG" if position > 0 else "SHORT"
    action = _stop_loss_action(side)

    active_stops = list(_iter_active_stop_orders(ib, ticker, action))
    if active_stops:
        first_order = getattr(active_stops[0], "order", None)
        stop_price = None
        if first_order is not None:
            stop_price = getattr(first_order, "auxPrice", None) or getattr(
                first_order, "lmtPrice", None
            )
        logger.info(
            "Existing stop-loss detected for %s (%s) at %s; no action taken.",
            ticker,
            side,
            f"{stop_price:.2f}" if isinstance(stop_price, (int, float)) else "unknown price",
        )
        return

    stop_loss_pct = _model_stop_loss_pct(model, ticker=ticker)
    if not stop_loss_pct:
        logger.warning(
            "No stop-loss percentage configured for %s using model %s; skipping.",
            ticker,
            model,
        )
        return

    entry_price = abs(avg_cost) if avg_cost else None
    if not entry_price:
        logger.warning("Unable to determine entry price for %s; skipping stop placement.", ticker)
        return

    stop_price = _compute_stop_loss_price(side, entry_price, stop_loss_pct)
    if stop_price is None:
        logger.warning(
            "Stop price could not be computed for %s (side=%s, entry=%.2f, pct=%s); skipping.",
            ticker,
            side,
            entry_price,
            stop_loss_pct,
        )
        return

    try:
        contract = Stock(ticker, "SMART", "USD")
        ib.qualifyContracts(contract)
        stop_price = round_to_min_tick(ib, contract, stop_price)
    except Exception as exc:  # pragma: no cover - defensive guard for connectivity issues
        logger.warning("Unable to round stop price for %s to min tick: %s", ticker, exc)

    success = _place_stop_loss_order(
        ib,
        ticker,
        side,
        abs(position),
        entry_price,
        stop_loss_pct,
    )
    if success:
        logger.info(
            "Placed stop-loss for %s (%s) at %.2f covering %s shares.",
            ticker,
            side,
            stop_price,
            abs(position),
        )
    else:
        logger.error("Failed to place stop-loss for %s (%s).", ticker, side)


def ensure_stop_losses(tickers: Iterable[str], model: str = "LSTM") -> None:
    ib = connect_ibkr()
    if ib is None or not ib.isConnected():
        logger.error("Unable to connect to IBKR; aborting stop-loss checks.")
        return

    for ticker in tickers:
        try:
            _ensure_stop_for_ticker(ib, ticker, model)
        except Exception as exc:  # pragma: no cover - defensive safety net
            logger.error("Unexpected error while processing %s: %s", ticker, exc)

    ib.disconnect()


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensure stop-loss orders exist for open positions.")
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Specific tickers to check. Defaults to the configured data/tickers.txt list.",
    )
    parser.add_argument(
        "--model",
        default="LSTM",
        help="Model name used to derive stop-loss percentages (default: LSTM).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    target_tickers = [ticker.upper() for ticker in args.tickers] if args.tickers else load_tickers()
    ensure_stop_losses(target_tickers, model=args.model)
