"""Centralized helpers for resolving project data paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

PathLike = Union[str, os.PathLike[str]]

_ROOT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA_DIR = _ROOT_DIR / "data"


def get_data_dir() -> Path:
    """Return the directory that stores datasets for the training pipeline.

    The location can be overridden by setting the ``AI_TRADING_DATA_DIR``
    environment variable. When unset, ``data/`` relative to the project root is
    used.
    """

    override = os.environ.get("AI_TRADING_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return _DEFAULT_DATA_DIR


def resolve_data_path(*parts: PathLike) -> Path:
    """Resolve *parts* inside the configured data directory."""

    return get_data_dir().joinpath(*map(Path, parts))
