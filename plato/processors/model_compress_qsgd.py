"""
Implements a Processor for compressing model weights.
"""
import logging
import pickle
from typing import Any
import sys
import zstd
import torch
import random
from struct import *

from plato.processors import model


class Processor(model.Processor):
    """
    Implements a Processor for compressing of model parameters.
    """

    def __init__(self, compression_level=1, **kwargs) -> None:
        super().__init__(**kwargs)

        self.compression_level = compression_level

    def process(self, data: Any) -> Any:
        """Implements a Processor for compressing model parameters."""

        data_size_0 = sys.getsizeof(pickle.dumps(data))
        data = super().process(data)
        data_size_1 = sys.getsizeof(pickle.dumps(data))

        output = zstd.compress(pickle.dumps(data), self.compression_level)
        data_size = sys.getsizeof(pickle.dumps(output))
        logging.info(
            "Compression level: %d, Original payload data size is %.2f MB, quantized payload data size is %.2f MB, compressed payload data size is %.2f MB (simulated).",
            self.compression_level,
            data_size_0 / 1024**2,
            data_size_1 / 1024**2,
            data_size / 1024**2,
        )
        # Compression level: 1, Original payload data size is 0.24 MB, quantized payload data size is 0.12 MB, compressed payload data size is 0.08 MB (simulated).

        if self.client_id is None:
            logging.info("[Server #%d] Compressed model parameters.", self.server_id)
        else:
            logging.info("[Client #%d] Compressed model parameters.", self.client_id)

        return output

    def _process_layer(self, layer: Any) -> Any:
        """Quantize and compress each individual layer of the model"""

        def add_prob(prob: Any) -> Any:
            size = prob.size()
            # Strange here! new_prob share same addresses with prob.
            new_prob = prob.reshape(-1)
            random.seed()
            for count, value in enumerate(new_prob):
                if random.random() <= value:
                    new_prob[count] = 1
                else:
                    new_prob[count] = 0
            return torch.reshape(new_prob, size).to(int)

        def handler(tensor: Any) -> Any:
            """Handler function for the compression of quantized parameter"""
            content = b""
            if len(tensor.size()) != 0:
                for i in tensor:
                    content += handler(i)
            else:
                num = int(tensor.item())
                if num < 0:
                    num = abs(num) ^ unpack("!i", b"\x00\x00\x00\x80")[0]
                content = pack("!I", num)[3:4]  # present a parameter in 1 byte
                # TODO: now only works for cases that quantization number <= 127
            return content

        # Step 1: quantization
        s = 64  # quantization level
        max_v = torch.max(abs(layer))  # the max absolute value
        ratio = abs(layer) / max_v  # |v_i| / ||v||
        l = (ratio * s - 1).ceil().to(int)
        prob = ratio * s - l  # probability for l+1 / s
        zeta = l.to(int) + add_prob(prob)  # add 1 for the hitted probability
        neg = ((-1) * layer.lt(0) + 1 * layer.ge(0)).to(
            int
        )  # -1 for negative, otherwise 1
        zeta = zeta.mul(neg)
        # print("zeta (with sign):")
        # print(zeta)

        # Step 2: handle head, including v_max, len() and their following dims
        header = pack("!f", max_v.item())  # ! represents for big-endian
        header += pack("!I", zeta.numel())
        dimensions = len(zeta.size())
        header += pack("!h", dimensions)
        for i in range(dimensions):
            header += pack("!h", zeta.size(i))
        # print("header in hex: \n" + header.hex())

        # Step 3: handle content, each includes 1 sign bit followed by 7 bits
        content = handler(zeta)
        # print("\ncontent in hex: \n" + content.hex())

        # Step 4: get the quantized and updated file
        output = header + content

        size_0 = sys.getsizeof(pickle.dumps(layer))  # original size
        size_1 = sys.getsizeof(
            pickle.dumps(layer.to(torch.bfloat16))
        )  # size of float16
        size_2 = sys.getsizeof(output)  # size of qsgd
        print(
            "parameter size of original, float16 and qsgd (in bytes): ",
            size_0,
            size_1,
            size_2,
        )  # size in bytes

        return output


# ./run -c examples/qfed/qfed_MNIST_lenet5.yml --cpu -b ./my_test
