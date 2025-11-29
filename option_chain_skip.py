"""List of tickers that should bypass option chain lookups."""

from __future__ import annotations

# Tickers known to lack option chain metadata from IBKR.
SKIP_OPTION_CHAIN_TICKERS: set[str] = {
    "FNGU",
    "PLTU",
    "QSU",
}


def should_skip_option_chain(ticker: str) -> bool:
    """Return ``True`` when a ticker should not query option chains.

    IBKR does not provide option chain metadata for a small set of symbols.
    Keeping the check here avoids downstream ``SyntaxError`` or parsing issues
    when stale or malformed files slip into a deployment image.
    """

    return ticker.upper() in SKIP_OPTION_CHAIN_TICKERS


__all__ = ["SKIP_OPTION_CHAIN_TICKERS", "should_skip_option_chain"]
