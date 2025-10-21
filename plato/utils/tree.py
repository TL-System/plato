"""
Utilities for flattening and reconstructing nested parameter trees.

These helpers are backend agnostic and operate on dictionaries, lists,
tuples, and leaf tensors/arrays. They are primarily used by serialization
layers to convert arbitrary nested structures into flat key-value maps for
transport or persistence, and restore them back when needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:  # pragma: no cover - optional dependency
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None


def _join_path(prefix: str, suffix: str) -> str:
    if not prefix:
        return suffix
    if suffix.startswith("["):
        return f"{prefix}{suffix}"
    return f"{prefix}.{suffix}"


def _index_path(prefix: str, index: int) -> str:
    return f"{prefix}[{index}]" if prefix else f"[{index}]"


def _ensure_numpy(value: Any) -> np.ndarray:
    """Convert backend tensors to numpy arrays."""
    if isinstance(value, np.ndarray):
        return value
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if mx is not None and isinstance(value, mx.array):
        if hasattr(mx, "to_numpy"):
            return mx.to_numpy(value)
        if hasattr(value, "to_numpy"):
            return value.to_numpy()
        if hasattr(value, "to_host"):
            return value.to_host()
        return np.array(value)
    if hasattr(value, "to_host"):
        return value.to_host()
    if hasattr(value, "__array__"):
        return np.asarray(value)
    return np.array(value)


def _detect_backend(value: Any) -> str:
    if torch is not None and isinstance(value, torch.Tensor):
        return "torch"
    if mx is not None and isinstance(value, mx.array):
        return "mlx"
    if isinstance(value, np.ndarray):
        return "numpy"
    return "native"


def _restore_backend(array: np.ndarray, backend: str | None) -> Any:
    if backend == "torch":
        if torch is None:
            raise ImportError("Torch is required to restore torch tensors.")
        return torch.from_numpy(array).clone()
    if backend == "mlx":
        return array
    return array


@dataclass
class TreeMetadata:
    """Metadata describing the structure of a flattened tree."""

    type: str
    children: List[str] | None = None
    container: str | None = None  # distinguish tuple vs list
    backend: str | None = None


def flatten_tree(tree: Any) -> Tuple[Dict[str, np.ndarray], Dict[str, TreeMetadata]]:
    """
    Flatten a nested tree into a dict of numpy arrays keyed by path segments.

    Returns:
        tuple(dict, dict): (flat leaf map, metadata describing the tree)
    """

    flat: Dict[str, np.ndarray] = {}
    metadata: Dict[str, TreeMetadata] = {}

    def recurse(node: Any, path: str) -> None:
        if isinstance(node, dict):
            metadata[path] = TreeMetadata(
                type="dict", children=list(node.keys()), container=None
            )
            for key, value in node.items():
                recurse(value, _join_path(path, key))
            return

        if isinstance(node, (list, tuple)):
            metadata[path] = TreeMetadata(
                type="sequence",
                children=[_index_path(path, idx) for idx in range(len(node))],
                container="tuple" if isinstance(node, tuple) else "list",
            )
            for idx, value in enumerate(node):
                recurse(value, _index_path(path, idx))
            return

        key = path or "__root__"
        backend = _detect_backend(node)
        metadata[path] = TreeMetadata(
            type="leaf", children=None, container=None, backend=backend
        )
        flat[key] = _ensure_numpy(node)

    recurse(tree, "")
    return flat, metadata


def unflatten_tree(
    flat: Dict[str, np.ndarray], metadata: Dict[str, TreeMetadata]
) -> Any:
    """Rebuild a nested tree from flattened leaves and metadata."""

    def build(path: str) -> Any:
        entry = metadata.get(path)
        if entry is None:
            raise KeyError(f"Missing metadata for path '{path}'.")

        if entry.type == "leaf":
            key = path or "__root__"
            if key not in flat:
                raise KeyError(f"Missing tensor data for leaf '{key}'.")
            return _restore_backend(flat[key], entry.backend)

        if entry.type == "dict":
            result = {}
            for key in entry.children or []:
                child_path = _join_path(path, key)
                result[key] = build(child_path)
            return result

        if entry.type == "sequence":
            items = []
            for idx, child in enumerate(entry.children or []):
                child_path = _index_path(path, idx) if path else child
                items.append(build(child_path))
            if entry.container == "tuple":
                return tuple(items)
            return items

        raise ValueError(f"Unsupported tree metadata type '{entry.type}'.")

    return build("")
