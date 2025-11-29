"""Deprecated rule-based trading entry points."""

import logging
from typing import Any

logger = logging.getLogger("rule_based_trader")


def trade_ema10_crossover(*_: Any, **__: Any) -> None:
    """Former EMA crossover rule has been retired."""

    logger.info(
        "Rule-based EMA10 crossover trading has been removed; no action will be taken."
    )


# Backwards compatibility for older imports
trade_cross_up_ema10 = trade_ema10_crossover
