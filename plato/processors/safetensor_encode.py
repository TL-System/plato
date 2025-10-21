"""
Processor that serialises nested payloads into Safetensors bytes.
"""

from __future__ import annotations

from typing import Any

from plato.processors import base
from plato.serialization.safetensor import serialize_tree


class Processor(base.Processor):
    """Serialise outbound payloads to Safetensors bytes."""

    def process(self, data: Any) -> bytes:
        if data is None:
            return b""
        return serialize_tree(data)
