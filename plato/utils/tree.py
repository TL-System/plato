"""
Utilities for flattening and reconstructing nested parameter trees.

These helpers are backend agnostic and operate on dictionaries, lists,
tuples, and leaf tensors/arrays. They are primarily used by serialization
layers to convert arbitrary nested structures into flat key-value maps for
transport or persistence, and restore them back when needed.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, cast

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import mlx.core as mx
    import torch
else:  # pragma: no cover - optional dependency
    try:
        torch = cast(ModuleType, importlib.import_module("torch"))
    except ImportError:
        torch = None

    try:
        mx = cast(ModuleType, importlib.import_module("mlx.core"))
    except ImportError:
        mx = None


_TORCH_TENSOR_TYPE = getattr(torch, "Tensor", None) if torch is not None else None
_MX_ARRAY_TYPE = getattr(mx, "array", None) if mx is not None else None


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
    if value is None:
        return np.zeros(0, dtype=np.uint8)
    if isinstance(value, str):
        return np.frombuffer(value.encode("utf-8"), dtype=np.uint8)
    if isinstance(value, (bytes, bytearray)):
        return np.frombuffer(bytes(value), dtype=np.uint8)
    if isinstance(value, np.ndarray):
        return value
    if _TORCH_TENSOR_TYPE is not None and isinstance(value, _TORCH_TENSOR_TYPE):
        tensor = value
        detach_fn = getattr(tensor, "detach", None)
        if callable(detach_fn):
            tensor = detach_fn()
        cpu_fn = getattr(tensor, "cpu", None)
        if callable(cpu_fn):
            tensor = cpu_fn()
        numpy_fn = getattr(tensor, "numpy", None)
        if callable(numpy_fn):
            return numpy_fn()
        return np.asarray(tensor)
    if _MX_ARRAY_TYPE is not None and isinstance(value, _MX_ARRAY_TYPE):
        to_numpy_fn = getattr(mx, "to_numpy", None) if mx is not None else None
        if callable(to_numpy_fn):
            return to_numpy_fn(value)
        for attr in ("to_numpy", "to_host"):
            candidate = getattr(value, attr, None)
            if callable(candidate):
                return candidate()
        return np.array(value)
    if hasattr(value, "to_host"):
        return value.to_host()
    if hasattr(value, "__array__"):
        return np.asarray(value)
    return np.array(value)


def _detect_backend(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (bytes, bytearray)):
        return "bytes"
    if _TORCH_TENSOR_TYPE is not None and isinstance(value, _TORCH_TENSOR_TYPE):
        return "torch"
    if _MX_ARRAY_TYPE is not None and isinstance(value, _MX_ARRAY_TYPE):
        return "mlx"
    if isinstance(value, np.ndarray):
        return "numpy"
    return "native"


def _restore_backend(array: np.ndarray, backend: str | None) -> Any:
    if backend == "none":
        return None
    if backend == "string":
        return array.tobytes().decode("utf-8")
    if backend == "bytes":
        return array.tobytes()
    if backend == "torch":
        if torch is None:
            raise ImportError("Torch is required to restore torch tensors.")
        from_numpy_fn = getattr(torch, "from_numpy", None)
        if not callable(from_numpy_fn):
            raise AttributeError("torch.from_numpy is unavailable.")
        tensor = from_numpy_fn(array)
        clone_fn = getattr(tensor, "clone", None)
        return clone_fn() if callable(clone_fn) else tensor
    if backend == "mlx":
        return array
    return array


@dataclass
class TreeMetadata:
    """Metadata describing the structure of a flattened tree."""

    type: str
    children: list[str] | None = None
    container: str | None = None  # distinguish tuple vs list
    backend: str | None = None


def flatten_tree(tree: Any) -> tuple[dict[str, np.ndarray], dict[str, TreeMetadata]]:
    """
    Flatten a nested tree into a dict of numpy arrays keyed by path segments.

    Returns:
        tuple(dict, dict): (flat leaf map, metadata describing the tree)
    """

    flat: dict[str, np.ndarray] = {}
    metadata: dict[str, TreeMetadata] = {}

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
    flat: dict[str, np.ndarray], metadata: dict[str, TreeMetadata]
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
