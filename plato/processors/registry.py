"""
This registry for Processors contains framework-specific implementations of
Processors for data payloads.

Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""
import logging
from collections import OrderedDict
from typing import Tuple

from plato.config import Config
from plato.processors import pipeline

if not (hasattr(Config().trainer, 'use_tensorflow')
        or hasattr(Config().trainer, 'use_mindspore')):
    from plato.processors import (base, mistnet_inbound_features,
                                  mistnet_outbound_features,
                                  mistnet_randomized_response,
                                  mistnet_gaussian, mistnet_laplace,
                                  mistnet_unbatch, mistnet_quantize,
                                  mistnet_dequantize)

    registered_processors = OrderedDict([
        ('base', base.Processor),
        ('mistnet_randomized_response', mistnet_randomized_response.Processor),
        ('mistnet_unbatch', mistnet_unbatch.Processor),
        ('mistnet_outbound_features', mistnet_outbound_features.Processor),
        ('mistnet_inbound_features', mistnet_inbound_features.Processor),
        ('mistnet_gaussian', mistnet_gaussian.Processor),
        ('mistnet_laplace', mistnet_laplace.Processor),
        ('mistnet_quantize', mistnet_quantize.Processor),
        ('mistnet_dequantize', mistnet_dequantize.Processor),
    ])


def get(user: str, *args,
        **kwargs) -> Tuple[pipeline.Processor, pipeline.Processor]:
    """ Get an instance of the processor. """
    outbound_processors = []
    inbound_processors = []

    assert user in ("Server", "Client")

    if user == "Server":
        config = Config().server
    else:
        config = Config().clients

    if hasattr(config, 'outbound_processors') and isinstance(
            config.outbound_processors, list):
        outbound_processors = config.outbound_processors

    if hasattr(config, 'inbound_processors') and isinstance(
            config.inbound_processors, list):
        inbound_processors = config.inbound_processors

    for processor in outbound_processors:
        logging.info("%s: Using Processor for sending payload: %s", user,
                     processor)
    for processor in inbound_processors:
        logging.info("%s: Using Processor for receiving payload: %s", user,
                     processor)

    outbound_processors = list(
        map(lambda name: registered_processors[name](*args, **kwargs),
            outbound_processors))
    inbound_processors = list(
        map(lambda name: registered_processors[name](*args, **kwargs),
            inbound_processors))

    return pipeline.Processor(outbound_processors), pipeline.Processor(
        inbound_processors)
