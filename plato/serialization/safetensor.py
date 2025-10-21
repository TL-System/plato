"""
Helpers for serialising nested parameter trees with Safetensors.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict

import numpy as np
from safetensors.numpy import load, save

from plato.utils.tree import TreeMetadata, flatten_tree, unflatten_tree


def _metadata_to_json(metadata: Dict[str, TreeMetadata]) -> str:
    serialisable = {
        path: {
            "type": value.type,
            "children": value.children,
            "container": value.container,
            "backend": value.backend,
        }
        for path, value in metadata.items()
    }
    return json.dumps(serialisable)


def _metadata_from_json(payload: str) -> Dict[str, TreeMetadata]:
    data = json.loads(payload)
    return {
        path: TreeMetadata(
            type=value["type"],
            children=value.get("children"),
            container=value.get("container"),
            backend=value.get("backend"),
        )
        for path, value in data.items()
    }


def serialize_tree(tree: Any) -> bytes:
    """
    Serialise a nested tree of tensors/arrays into Safetensors bytes.
    """
    flat, metadata = flatten_tree(tree)
    tensors = {path or "__root__": array for path, array in flat.items()}
    metadata_json = _metadata_to_json(metadata).encode("utf-8")
    tensors["_tree_metadata"] = np.frombuffer(metadata_json, dtype=np.uint8)
    return save(tensors)


def deserialize_tree(buffer: bytes | bytearray | memoryview) -> Any:
    """
    Restore a nested tree of numpy arrays from Safetensors bytes.
    """
    if isinstance(buffer, (bytearray, memoryview)):
        buffer = bytes(buffer)
    tensors = load(buffer)
    metadata_blob = tensors.pop("_tree_metadata", None)

    if metadata_blob is None:
        # Fallback: treat as a flat dict
        return tensors

    metadata_json = metadata_blob.tobytes().decode("utf-8")
    metadata = _metadata_from_json(metadata_json)
    flat = {("" if key == "__root__" else key): value for key, value in tensors.items()}
    return unflatten_tree(flat, metadata)
