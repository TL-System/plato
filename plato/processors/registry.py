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
    from plato.processors import (
        base,
        feature_randomized_response,
        feature_gaussian,
        feature_laplace,
        feature_quantize,
        feature_dequantize,
        feature_unbatch,
        inbound_feature_tensors,
        outbound_feature_ndarrays,
        model_deepcopy,
        model_quantize,
        model_dequantize,
        model_randomized_response,
    )

    registered_processors = OrderedDict([
        ('base', base.Processor),
        ('feature_randomized_response', feature_randomized_response.Processor),
        ('feature_gaussian', feature_gaussian.Processor),
        ('feature_laplace', feature_laplace.Processor),
        ('feature_quantize', feature_quantize.Processor),
        ('feature_dequantize', feature_dequantize.Processor),
        ('feature_unbatch', feature_unbatch.Processor),
        ('inbound_feature_tensors', inbound_feature_tensors.Processor),
        ('outbound_feature_ndarrays', outbound_feature_ndarrays.Processor),
        ('model_deepcopy', model_deepcopy.Processor),
        ('model_quantize', model_quantize.Processor),
        ('model_dequantize', model_dequantize.Processor),
        ('model_randomized_response', model_randomized_response.Processor),
    ])


def get(user: str,
        processor_kwargs={},
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

    def map_f(name):
        if name in processor_kwargs:
            this_kwargs = {**kwargs, **(processor_kwargs[name])}
        else:
            this_kwargs = kwargs

        return registered_processors[name](**this_kwargs)

    outbound_processors = list(map(map_f, outbound_processors))
    inbound_processors = list(map(map_f, inbound_processors))

    return pipeline.Processor(outbound_processors), pipeline.Processor(
        inbound_processors)
