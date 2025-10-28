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
    """Return the root directory of the Nanochat submodule."""
    repo_root = Path(__file__).resolve().parents[2]
    nanochat_root = repo_root / "external" / "nanochat"
    if not nanochat_root.exists():
        raise ThirdPartyImportError(
            "Nanochat submodule missing. Run `git submodule update --init --recursive`."
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
