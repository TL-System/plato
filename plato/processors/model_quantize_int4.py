"""
Implements a Processor to quantize and pack upload models into INT4 representation.

Floating-point parameters are quantized to signed 4-bit integers and packed
two per byte, reducing transmission payload size by approximately 8x relative to float32.
"""

from struct import pack
from typing import Any

import numpy as np
import torch

from plato.processors import model


class Processor(model.Processor):
    """
    Implements a Processor to quantize model parameters into INT4 nibbles.
    """

    def _process_layer(self, layer: Any) -> Any:
        """Quantizes each individual layer of the model with INT4."""
        if not isinstance(layer, torch.Tensor) or not layer.is_floating_point():
            return layer

        if not torch.isfinite(layer).all():
            raise ValueError("Cannot quantize layer containing NaN or Inf values.")

        numel = layer.numel()
        dims = len(layer.shape)

        if numel == 0:
            header = pack("!fIh", 0.0, 0, dims)
            for d in layer.shape:
                header += pack("!h", d)
            return header

        max_v = float(torch.max(torch.abs(layer)).item())
        header = pack("!fIh", max_v, numel, dims)
        for d in layer.shape:
            header += pack("!h", d)

        flat = layer.detach().flatten().float()
        if max_v == 0.0:
            quantized = torch.zeros_like(flat, dtype=torch.int32)
        else:
            scale = max_v / 7.0
            quantized = torch.clamp(torch.round(flat / scale), -8, 7).to(torch.int32)

        nibbles = (quantized + 8).to(torch.uint8)

        if nibbles.numel() % 2 != 0:
            nibbles = torch.nn.functional.pad(nibbles, (0, 1), value=8)

        paired = nibbles.view(-1, 2)
        packed = (
            ((paired[:, 0] << 4) | (paired[:, 1] & 0x0F)).cpu().numpy().astype(np.uint8)
        )

        return header + packed.tobytes()
