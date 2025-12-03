"""Compatibility shim for backtesting scripts.

This module exposes the ``connect_ibkr`` helper used by live trading
components. The implementation is shared with ``tsla_ai_master_final_ready``
so that backtesting utilities can import ``live_trading`` without depending
on notebook-only names or alternative entry points.
"""

from tsla_ai_master_final_ready import connect_ibkr

__all__ = ["connect_ibkr"]
