"""Helpers for scripts executed as standalone modules.

This module ensures that project-local imports (e.g. ``csv_utils``) resolve when
invoking scripts from arbitrary working directories such as ``/app`` inside the
Docker runtime.
"""

from __future__ import annotations

from pathlib import Path
import sys


def ensure_project_root_on_path() -> Path:
    """Ensure the repository root is present on ``sys.path``.

    Returns the resolved project root path so callers can reuse it when they
    need to locate resources relative to the repository.
    """

    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


__all__ = ["ensure_project_root_on_path"]
