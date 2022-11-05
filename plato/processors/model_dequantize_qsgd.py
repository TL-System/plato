"""
Implements a Processor to decompress and dequantize upload models.

In more detail, this processor first decompress each received parameter. Next,
dequantize quantize each upload parameter under given quantization level.
Hence, the 8-bit received parameter can be transfered to 32-bit parameter.

Reference:

Alistarh, D., Grubic, D., Li, J., Tomioka, R., & Vojnovic, M. (2017).
"QSGD: Communication-efficient SGD via gradient quantization and encoding."
Advances in neural information processing systems, 30.

https://proceedings.neurips.cc/paper/2017/file/6c340f25839e6acdc73414517203f5f0-Paper.pdf
"""

import logging
from typing import Any
import pickle
from struct import unpack
import sys
import torch

from plato.processors import model


class Processor(model.Processor):
    """
    Implements a Processor for decompressing of model parameters.
    """

    def __init__(self, quantization_level=64, **kwargs) -> None:
        super().__init__(**kwargs)

        self.quantization_level = quantization_level  # must <= 128!

    def process(self, data: Any) -> Any:
        """Implements a Processor for decompressing model parameters."""

        data_size_old = sys.getsizeof(pickle.dumps(data))
        output = super().process(data)
        data_size_new = sys.getsizeof(pickle.dumps(output))

        if self.client_id is None:
            logging.info(
                "[Server #%d] Dequantized and decompressed received upload model parameters.",
                self.server_id,
            )
            logging.info(
                "[Server #%d] Quantization level: %d, received payload data size is %.2f MB,"
                "dequantized size is %.2f MB(simulated).",
                self.server_id,
                self.quantization_level,
                data_size_old / 1024**2,
                data_size_new / 1024**2,
            )
        else:
            logging.info(
                "[Client #%d] Dequantized received model parameters.", self.client_id
            )
        return output

    def _process_layer(self, layer: Any) -> Any:
        """Decompress and dequantize each individual layer of the model"""

        # Step 1: decompress the header
        tuning_param = self.quantization_level - 1  # quantization level
        max_v = unpack("!f", layer[0:4])[0]
        numel = unpack("!I", layer[4:8])[0]
        dimensions = unpack("!h", layer[8:10])[0]
        size = []
        for i in range(dimensions):
            size.append(unpack("!h", layer[10 + 2 * i : 12 + 2 * i])[0])

        # Step 2: decompress the content
        layer = layer[10 + 2 * dimensions :]
        zeta = []
        prefix = b"\x00\x00\x00"
        for i in range(numel):
            tmp = unpack("!I", prefix + layer[i : i + 1])[0]
            if tmp >= 128:
                tmp = -1 * (tmp - 128)
            zeta.append(tmp)
        zeta = torch.tensor(zeta).reshape(size)

        # Step 3: dequantize the content
        zeta = zeta * max_v / tuning_param

        return zeta
