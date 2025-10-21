"""
Processor that deserialises Safetensors bytes back into nested payloads.
"""

from __future__ import annotations

from typing import Any, Union

from plato.processors import base
from plato.serialization.safetensor import deserialize_tree


class Processor(base.Processor):
    """Decode Safetensors bytes into numpy-backed payloads."""

    def process(self, data: Union[bytes, bytearray, memoryview, Any]) -> Any:
        if data in (None, b"", bytearray()):
            return None
        if isinstance(data, (bytes, bytearray, memoryview)):
            buffer = bytes(data)
        else:
            buffer = data
            if not isinstance(buffer, (bytes, bytearray, memoryview)):
                raise TypeError(
                    "Safetensor decode processor expects raw bytes payload."
                )
            buffer = bytes(buffer)
        return deserialize_tree(buffer)
