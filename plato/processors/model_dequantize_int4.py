"""
Implements a Processor to unpack and dequantize models from INT4 representation.
"""

from struct import unpack
from typing import Any

import numpy as np
import torch

from plato.processors import model


class Processor(model.Processor):
    """
    Implements a Processor to dequantize model parameters quantized with INT4.
    """

    def _process_layer(self, layer: Any) -> Any:
        """Dequantizes each individual layer of the model from INT4 nibbles."""
        if not isinstance(layer, (bytes, bytearray)):
            return layer

        if len(layer) < 10:
            raise ValueError("Truncated INT4 header.")

        max_v, numel, dims = unpack("!fIh", layer[:10])
        if not np.isfinite(max_v):
            raise ValueError("Non-finite scale in INT4 header.")

        offset = 10
        shape = []
        for _ in range(dims):
            shape.append(unpack("!h", layer[offset : offset + 2])[0])
            offset += 2

        if numel == 0:
            return torch.empty(shape, dtype=torch.float32)

        if max_v == 0.0:
            return torch.zeros(shape, dtype=torch.float32)

        body = layer[offset:]
        expected_bytes = (numel + 1) // 2
        if len(body) != expected_bytes:
            raise ValueError("Payload length does not match INT4 element count.")

        packed_arr = np.frombuffer(body, dtype=np.uint8).copy()
        packed_t = torch.from_numpy(packed_arr)

        high = (packed_t >> 4) & 0x0F
        low = packed_t & 0x0F
        unpacked = torch.stack([high, low], dim=1).flatten()[:numel]

        scale = max_v / 7.0
        restored = (unpacked.float() - 8.0) * scale
        if not torch.isfinite(restored).all():
            raise ValueError("Decoded non-finite values in INT4 dequantization.")

        return restored.reshape(shape)
