"""
Utility helpers for working with optional zstd compression backends.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType
from typing import Any, Final, cast

_CANDIDATES: Final[tuple[str, ...]] = ("zstd", "zstandard")
_MODULE: ModuleType | None = None


def _load() -> ModuleType:
    """Try to import a supported zstd-compatible module."""
    for name in _CANDIDATES:
        if importlib.util.find_spec(name) is None:
            continue
        module = importlib.import_module(name)
        return module
    raise ModuleNotFoundError(
        "Missing zstd compression backend. Install either `zstd` or `zstandard`."
    )


def get_zstd() -> Any:
    """Return the loaded zstd-compatible module."""
    global _MODULE  # noqa: PLW0603 - cache module for reuse
    if _MODULE is None:
        _MODULE = _load()
    return cast(Any, _MODULE)
