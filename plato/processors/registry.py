"""
This registry for Processors contains framework-specific implementations of
Processors for data payloads.

Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""

import logging
from typing import Tuple

from plato.config import Config
from plato.processors import (
    base,
    compress,
    decompress,
    model_compress,
    model_decompress,
    model_deepcopy,
    model_dequantize,
    model_dequantize_qsgd,
    model_quantize,
    model_quantize_qsgd,
    model_randomized_response,
    mpc_model_encrypt_additive,
    mpc_model_encrypt_shamir,
    pipeline,
    structured_pruning,
    unstructured_pruning,
)

registered_processors = {
    "base": base.Processor,
    "compress": compress.Processor,
    "decompress": decompress.Processor,
    "model_deepcopy": model_deepcopy.Processor,
    "model_quantize": model_quantize.Processor,
    "model_dequantize": model_dequantize.Processor,
    "model_compress": model_compress.Processor,
    "model_quantize_qsgd": model_quantize_qsgd.Processor,
    "model_decompress": model_decompress.Processor,
    "model_dequantize_qsgd": model_dequantize_qsgd.Processor,
    "model_randomized_response": model_randomized_response.Processor,
    "structured_pruning": structured_pruning.Processor,
    "unstructured_pruning": unstructured_pruning.Processor,
    "mpc_model_encrypt_additive": mpc_model_encrypt_additive.Processor,
    "mpc_model_encrypt_shamir": mpc_model_encrypt_shamir.Processor,
}


def register_he_processors():
    """Register homomorphic encryption processors if needed."""


def get(
    user: str, processor_kwargs=None, **kwargs
) -> Tuple[pipeline.Processor, pipeline.Processor]:
    """Get an instance of the processor."""
    outbound_processors = []
    inbound_processors = []

    assert user in ("Server", "Client")

    if user == "Server":
        config = Config().server
    else:
        config = Config().clients

    if hasattr(config, "outbound_processors") and isinstance(
        config.outbound_processors, list
    ):
        outbound_processors = config.outbound_processors

    if hasattr(config, "inbound_processors") and isinstance(
        config.inbound_processors, list
    ):
        inbound_processors = config.inbound_processors

    for processor in outbound_processors:
        logging.info("%s: Using Processor for sending payload: %s", user, processor)
    for processor in inbound_processors:
        logging.info("%s: Using Processor for receiving payload: %s", user, processor)

    # Check if HE processors are needed based on server configuration
    if hasattr(config, "type") and config.type == "fedavg_he":
        # FedAvg server with homomorphic encryption needs to import tenseal,
        # which is not available on all platforms such as macOS
        from plato.processors import model_decrypt, model_encrypt

        registered_processors.update(
            {
                "model_encrypt": model_encrypt.Processor,
                "model_decrypt": model_decrypt.Processor,
            }
        )

        logging.info("%s: Using homomorphic encryption processors.", user)

    def map_f(name):
        if processor_kwargs is not None and name in processor_kwargs:
            this_kwargs = {**kwargs, **(processor_kwargs[name])}
        else:
            this_kwargs = kwargs

        return registered_processors[name](name=name, **this_kwargs)

    outbound_processors = list(map(map_f, outbound_processors))
    inbound_processors = list(map(map_f, inbound_processors))

    return pipeline.Processor(outbound_processors), pipeline.Processor(
        inbound_processors
    )
