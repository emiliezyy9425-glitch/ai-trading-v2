import re
import math
from typing import Any


def format_duration(num: int, unit: str) -> str:
    """Return an IBKR duration string with a space before the unit."""

    return f"{num} {unit.upper()}"


def get_duration_for_timeframe(timeframe: str, default: str = "10 Y") -> str:
    """Return a properly spaced duration string for an IBKR timeframe.

    Parameters
    ----------
    timeframe : str
        Human-readable timeframe (e.g. "1 hour", "4 hours", "1 day").
    default : str, optional
        Fallback duration if *timeframe* has no explicit mapping.

    Returns
    -------
    str
        Duration string formatted as ``"<int> <unit>"`` where unit is one of
        ``S``, ``D``, ``W``, ``M`` or ``Y``.
    """
    mapping = {
        "1 hour": format_duration(180, "D"),  # ~6 months
        "4 hours": format_duration(365, "D"),  # ~1 year
        "1 day": format_duration(3650, "D"),   # ~10 years
    }
    duration = mapping.get(timeframe, default)
    match = re.match(r"(\d+)\s*([a-zA-Z]+)", duration.strip())
    if not match:
        return duration
    num, unit = match.groups()
    return format_duration(int(num), unit)


def round_to_min_tick(ib: Any, contract: Any, price: float, default: float = 0.01) -> float:
    """Round ``price`` to the contract's minimum tick size.

    If the contract exposes a tick table (``priceIncrements``) the increment
    applicable to *price* is used; otherwise ``minTick`` or *default* is
    applied.

    Parameters
    ----------
    ib : object
        IB instance used to query contract details. If it lacks
        ``reqContractDetails`` or the call fails, *default* is used.
    contract : object
        Contract for which to determine the tick size.
    price : float
        The price to round.
    default : float, optional
        Fallback minimum tick if details are unavailable.

    Returns
    -------
    float
        ``price`` rounded to the nearest multiple of the minimum tick.
    """

    min_tick = default

    # First try to fetch tick info from IBKR if available
    if hasattr(ib, "reqContractDetails"):
        try:
            details = ib.reqContractDetails(contract)
            if details:
                detail = details[0]
                increments = (
                    getattr(detail, "priceIncrements", None)
                    or getattr(detail, "priceIncrement", None)
                )

                if isinstance(increments, (list, tuple)) and increments:
                    min_tick = default
                    key_fn = lambda x: getattr(
                        x, "lowEdge", x[0] if isinstance(x, (list, tuple)) else 0
                    )
                    for inc in sorted(increments, key=key_fn):
                        low = getattr(
                            inc,
                            "lowEdge",
                            inc[0] if isinstance(inc, (list, tuple)) else 0,
                        )
                        tick = getattr(
                            inc,
                            "increment",
                            inc[1] if isinstance(inc, (list, tuple)) else default,
                        )
                        if price >= low:
                            min_tick = tick
                        else:
                            break
                else:
                    min_tick = getattr(detail, "minTick", default) or default
        except Exception:
            pass

    # Manual fallback for Hong Kong Exchange where minTick often reports 0.01
    exchange = getattr(contract, "exchange", "").upper()
    if exchange == "SEHK":
        # Hong Kong Stock Exchange tick table
        hk_ticks = [
            (0, 0.25, 0.001),
            (0.25, 0.5, 0.005),
            (0.5, 10, 0.01),
            (10, 20, 0.02),
            (20, 100, 0.05),
            (100, 200, 0.1),
            (200, 500, 0.2),
            (500, 1000, 0.5),
            (1000, 2000, 1),
            (2000, 5000, 2),
            (5000, 9995, 5),
        ]
        for low, high, tick in hk_ticks:
            if low <= price < high:
                min_tick = tick
                break

    precision = max(int(math.ceil(-math.log10(min_tick))), 0)
    return round(round(price / min_tick) * min_tick, precision)
