"""
Implements a Processor to decompress and dequantize upload models.

In more detail, this processor first decompresses each received parameter.
Next, dequantize each upload parameter under the given quantization level.
Hence, 8-bit received parameters can be converted into 32-bit parameters.

Reference:

Alistarh, D., Grubic, D., Li, J., Tomioka, R., & Vojnovic, M. (2017).
"QSGD: Communication-efficient SGD via gradient quantization and encoding."
Advances in neural information processing systems.

https://proceedings.neurips.cc/paper/2017/file/6c340f25839e6acdc73414517203f5f0-Paper.pdf
"""

from typing import Any
from struct import unpack
import torch

from plato.processors import model


class Processor(model.Processor):
    """
    Implements a Processor to dequantize model parameters quantized with QSGD.
    """

    def __init__(self, quantization_level=64, **kwargs) -> None:
        super().__init__(**kwargs)

        self.quantization_level = quantization_level  # must <= 128!

    def _process_layer(self, layer: Any) -> Any:
        """Dequantizes each individual layer of the model."""

        # Step 1: decompress the header
        tuning_param = self.quantization_level - 1
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
