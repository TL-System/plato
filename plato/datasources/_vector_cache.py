"""Validated, atomic NPZ publication for Purchase and Texas dataset caches."""

import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np


class InvalidVectorCacheError(ValueError):
    """A vector cache is incomplete or has incompatible features and labels."""


def _validate(features: np.ndarray, labels: np.ndarray) -> None:
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise InvalidVectorCacheError(
            "Dataset cache needs a feature matrix and matching one-dimensional labels."
        )


def load_cache(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate all arrays; callers hold the dataset preparation lock."""
    try:
        dataset = np.load(path, allow_pickle=False)
        if not isinstance(dataset, np.lib.npyio.NpzFile):
            raise InvalidVectorCacheError("Dataset cache must be an NPZ archive.")
        with dataset:
            features, labels = dataset["X"], dataset["Y"]
    except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise InvalidVectorCacheError(f"Invalid dataset cache {path}: {exc}") from exc
    _validate(features, labels)
    return features, labels


def discard_abandoned_writes(path: str | Path) -> None:
    """Remove private temporary files after acquiring the preparation lock."""
    path = Path(path)
    for temporary in path.parent.glob(f".{path.name}.*.tmp"):
        temporary.unlink()


def publish_cache(path: str | Path, features: np.ndarray, labels: np.ndarray) -> None:
    """Publish a complete NPZ by atomic replacement within its filesystem."""
    _validate(features, labels)
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            np.savez(output, X=features, Y=labels)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
