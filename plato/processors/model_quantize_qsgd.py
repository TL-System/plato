"""
Implements a Processor to quantize and compress upload models.

In more detail, this processor first quantizes each upload parameter under
the given quantization level. Next, compress and store each quantized value.
Hence, these 32-bit parameters can be converted into 8-bit parameters.

Reference:

Alistarh, D., Grubic, D., Li, J., Tomioka, R., & Vojnovic, M. (2017).
"QSGD: Communication-efficient SGD via gradient quantization and encoding."
Advances in neural information processing systems.

https://proceedings.neurips.cc/paper/2017/file/6c340f25839e6acdc73414517203f5f0-Paper.pdf
"""

from typing import Any
import random
from struct import pack, unpack
import torch

from plato.processors import model


class Processor(model.Processor):
    """
    Implements a Processor to quantize model parameters with QSGD.
    """

    def __init__(self, quantization_level=64, **kwargs) -> None:
        super().__init__(**kwargs)

        self.quantization_level = quantization_level  # must <= 128!

    def _process_layer(self, layer: Any) -> Any:
        """Quantizes each individual layer of the model with QSGD."""

        def add_prob(prob: Any) -> Any:
            """Adds 1 to the corresponding positions with given probability."""
            size = prob.size()
            prob = prob.reshape(-1)
            random.seed()
            for count, value in enumerate(prob):
                if random.random() <= value:
                    prob[count] = 1
                else:
                    prob[count] = 0
            return torch.reshape(prob, size)

        def handler(tensor: Any) -> Any:
            """Handler function for the compression of quantized values."""
            content = b""
            tensor = tensor.reshape(-1)
            for _, value in enumerate(tensor):
                num = value.item()
                if num < 0:
                    num = abs(num) ^ unpack("!i", b"\x00\x00\x00\x80")[0]
                content += pack("!I", num)[3:4]  # present each parameter in 1 byte
            return content

        # Step 1: quantization
        tuning_param = self.quantization_level - 1  # tuning parameter
        max_v = torch.max(abs(layer))  # max absolute value
        neg = (-1) * layer.lt(0) + 1 * layer.ge(0)
        ratio = abs(layer) / max_v  # |v_i| / ||v||
        level = (ratio * tuning_param - 1).ceil()
        zeta = level + add_prob(ratio * tuning_param - level)
        zeta = zeta.mul(neg).to(int)

        # Step 2: handle the header
        output = pack("!f", max_v.item())  # ! represents for big-endian
        output += pack("!I", zeta.numel())
        dimensions = len(zeta.size())
        output += pack("!h", dimensions)
        for i in range(dimensions):
            output += pack("!h", zeta.size(i))

        # Step 3: handle the content, each consists of 1 sign bit followed by 7 bits
        output += handler(zeta)

        return output
