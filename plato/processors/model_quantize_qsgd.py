"""
Implements a Processor to quantize and compress upload models.

In more detail, this processor first quantize each upload parameter under given
quantization level. Next, compress and store each quantization value. Hence, the
32-bit parameter can be transfered to 8-bit parameter.

Reference:

Alistarh, D., Grubic, D., Li, J., Tomioka, R., & Vojnovic, M. (2017).
"QSGD: Communication-efficient SGD via gradient quantization and encoding."
Advances in neural information processing systems, 30.

https://proceedings.neurips.cc/paper/2017/file/6c340f25839e6acdc73414517203f5f0-Paper.pdf
"""
import logging
import pickle
from typing import Any
import sys
import random
from struct import pack, unpack
import torch

from plato.processors import model


class Processor(model.Processor):
    """
    Implements a Processor for compressing of model parameters.
    """

    def __init__(self, quantization_level=64, **kwargs) -> None:
        super().__init__(**kwargs)

        self.quantization_level = quantization_level  # must <= 128!

    def process(self, data: Any) -> Any:
        """Implements a Processor for compressing model parameters."""

        data_size_old = sys.getsizeof(pickle.dumps(data))
        output = super().process(data)
        data_size_new = sys.getsizeof(pickle.dumps(output))

        if self.client_id is None:
            logging.info("[Server #%d] Quantized model parameters.", self.server_id)
        else:
            logging.info(
                "[Client #%d] Quantized and compressed upload model parameters.",
                self.client_id,
            )
            logging.info(
                "[Client #%d] Quantization level: %d, original payload data size is %.2f MB,"
                "quantized size is %.2f MB (simulated).",
                self.client_id,
                self.quantization_level,
                data_size_old / 1024**2,
                data_size_new / 1024**2,
            )

        return output

    def _process_layer(self, layer: Any) -> Any:
        """Quantize and compress each individual layer of the model"""

        def add_prob(prob: Any) -> Any:
            """Add 1 to the corresponding positions with given probability."""
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
            """Handler function for the compression of quantized parameter"""
            content = b""
            tensor = tensor.reshape(-1)
            for _, value in enumerate(tensor):
                num = value.item()
                if num < 0:
                    num = abs(num) ^ unpack("!i", b"\x00\x00\x00\x80")[0]
                content += pack("!I", num)[3:4]  # present a parameter in 1 byte
            return content

        # Step 1: quantization
        tuning_param = self.quantization_level - 1  # tuning parameter
        max_v = torch.max(abs(layer))  # max absolute value
        neg = (-1) * layer.lt(0) + 1 * layer.ge(0)
        ratio = abs(layer) / max_v  # |v_i| / ||v||
        level = (ratio * tuning_param - 1).ceil()
        zeta = level + add_prob(ratio * tuning_param - level)
        zeta = zeta.mul(neg).to(int)

        # Step 2: handle head, including v_max, len() and their following dims
        output = pack("!f", max_v.item())  # ! represents for big-endian
        output += pack("!I", zeta.numel())
        dimensions = len(zeta.size())
        output += pack("!h", dimensions)
        for i in range(dimensions):
            output += pack("!h", zeta.size(i))

        # Step 3: handle content, each includes 1 sign bit followed by 7 bits
        output += handler(zeta)

        return output
