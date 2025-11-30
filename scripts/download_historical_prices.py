"""Placeholder market context fetchers."""

def fetch_iv_delta_spx(ib=None):
    """Return a neutral market context payload.

    The real implementation enriches option decisions with IV and delta data.
    Here we provide safe defaults so downstream logic can proceed without
    additional data dependencies.
    """

    return {
        "iv_atm": 0.0,
        "delta_atm_call": 0.0,
        "spx_above_20d": None,
        "per_ticker": {},
    }
