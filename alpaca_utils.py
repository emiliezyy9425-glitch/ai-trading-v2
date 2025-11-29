"""Utilities for working with Alpaca API credentials.

This module centralizes the logic for retrieving Alpaca credentials from the
environment.  Several scripts in this project historically relied on the
``ALPACA_API_KEY``/``ALPACA_SECRET_KEY`` environment variables, while other
Alpaca tooling (and the official documentation) often refers to the
``APCA_API_KEY_ID``/``APCA_API_SECRET_KEY`` pair.  In practice users may supply
either naming convention which previously caused the application to log
spurious warnings about missing credentials.  By handling the aliases in a
single place we ensure consistent behaviour across the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

__all__ = ["AlpacaCredentials", "get_alpaca_credentials", "have_alpaca_credentials"]


@dataclass(frozen=True)
class AlpacaCredentials:
    """Container for Alpaca API credentials and their source environment vars."""

    api_key: str
    secret_key: str
    api_key_env: str
    secret_key_env: str


# Prefer exact pairs first to avoid accidentally mixing production and paper
# credentials if both are present with different names.  The aliases cover the
# naming conventions encountered in the project as well as the defaults from
# Alpaca's own tooling.
_CREDENTIAL_ENV_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"),
    ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"),
    ("APCA_API_KEY", "APCA_API_SECRET_KEY"),
)

_API_KEY_ALIASES: Tuple[str, ...] = (
    "ALPACA_API_KEY",
    "APCA_API_KEY_ID",
    "APCA_API_KEY",
)

_SECRET_KEY_ALIASES: Tuple[str, ...] = (
    "ALPACA_SECRET_KEY",
    "APCA_API_SECRET_KEY",
    "APCA_API_SECRET",
)


def _first_env_with_value(names: Sequence[str]) -> tuple[Optional[str], Optional[str]]:
    """Return the first environment variable (name, value) pair that is set."""

    for name in names:
        value = os.getenv(name)
        if value:
            return value, name
    return None, None


def get_alpaca_credentials() -> Optional[AlpacaCredentials]:
    """Return Alpaca credentials from the environment if they are configured.

    The helper checks multiple environment variable aliases and returns the
    first matching pair.  ``None`` is returned when no complete credential set
    is configured.
    """

    for api_env, secret_env in _CREDENTIAL_ENV_PAIRS:
        api_key = os.getenv(api_env)
        secret_key = os.getenv(secret_env)
        if api_key and secret_key:
            return AlpacaCredentials(api_key=api_key, secret_key=secret_key, api_key_env=api_env, secret_key_env=secret_env)

    api_key, api_env = _first_env_with_value(_API_KEY_ALIASES)
    secret_key, secret_env = _first_env_with_value(_SECRET_KEY_ALIASES)
    if api_key and secret_key and api_env and secret_env:
        return AlpacaCredentials(api_key=api_key, secret_key=secret_key, api_key_env=api_env, secret_key_env=secret_env)
    return None


def have_alpaca_credentials() -> bool:
    """Return ``True`` when Alpaca credentials are available in the environment."""

    return get_alpaca_credentials() is not None
