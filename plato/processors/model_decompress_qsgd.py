"""
Implements a Processor for decompressing model weights.
"""

import logging
from typing import Any
import pickle
import zstd
from struct import *
import sys
import torch

from plato.processors import model


class Processor(model.Processor):
    """
    Implements a Processor for decompressing of model parameters.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def process(self, data: Any) -> Any:
        """Implements a Processor for decompressing model parameters."""

        output = pickle.loads(zstd.decompress(data))
        output = super().process(output)

        if self.client_id is None:
            logging.info(
                "[Server #%d] Decompressed received model parameters.", self.server_id
            )
        else:
            logging.info(
                "[Client #%d] Decompressed received model parameters.", self.client_id
            )
        return output

    def _process_layer(self, layer: Any) -> Any:
        """Decompress and dequantize each individual layer of the model"""

        # Step 1: decompress the header
        s = 64  # quantization level
        max_v = unpack("!f", layer[0:4])[0]
        numel = unpack("!I", layer[4:8])[0]
        dimensions = unpack("!h", layer[8:10])[0]
        size = []
        for i in range(dimensions):
            size.append(unpack("!h", layer[10 + 2 * i : 12 + 2 * i])[0])
        # print(max_v, numel, dimensions)

        # Step 2: decompress the content
        layer = layer[10 + 2 * dimensions :]
        # print(layer[0]) # 5
        # print(layer[0:1]) # b'\x05'
        zeta = []
        prefix = b"\x00\x00\x00"
        for i in range(numel):
            tmp = unpack("!I", prefix + layer[i : i + 1])[0]
            if tmp >= 128:
                tmp = -1 * (tmp - 128)
            zeta.append(tmp)
        zeta = torch.tensor(zeta)
        zeta = zeta.reshape(size)

        # Step 3: dequantize the content
        zeta = zeta * max_v / s
        # print(zeta)
        return zeta
