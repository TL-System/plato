"""
Helpers for accessing vendored third-party projects.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path


class ThirdPartyImportError(ImportError):
    """Raised when a vendored third-party project is unavailable."""


@lru_cache(maxsize=None)
def _nanochat_root() -> Path:
    """Return the root directory of the vendored Nanochat project."""
    repo_root = Path(__file__).resolve().parents[2]
    nanochat_root = repo_root / "runtime" / "third_party" / "nanochat"
    if not nanochat_root.exists():
        raise ThirdPartyImportError(
            "Nanochat is not vendored under runtime/third_party/nanochat."
        )
    return nanochat_root


def ensure_nanochat_importable() -> Path:
    """
    Ensure the vendored Nanochat package is importable.

    Returns:
        Path to the Nanochat project root.

    Raises:
        ThirdPartyImportError: If the vendored Nanochat tree is missing.
    """
    nanochat_root = _nanochat_root()
    path_str = str(nanochat_root)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return nanochat_root
