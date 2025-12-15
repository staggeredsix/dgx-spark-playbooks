"""Utility helpers for the MCP hub."""

from __future__ import annotations

import importlib
from typing import Any


def require(pkg: str, hint: str) -> Any:
    """Import a dependency lazily with an actionable error message."""

    try:
        return importlib.import_module(pkg)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            f"Missing dependency: {pkg}. Install {hint}. Root error: {exc}"
        ) from exc

