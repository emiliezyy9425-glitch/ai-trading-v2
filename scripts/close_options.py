"""Lightweight placeholder for option-closing utilities."""

def close_all_option_positions(ib, tickers=None, *, log_callback=None):
    """Stub implementation that reports no positions closed.

    The primary runtime relies on a richer implementation housed outside this
    environment. This placeholder keeps dependent modules importable and
    returns a neutral result indicating that no action was taken.
    """

    result = {
        "closed": [],
        "errors": [],
    }
    if log_callback:
        log_callback("No option positions closed (stub implementation).")
    return result
