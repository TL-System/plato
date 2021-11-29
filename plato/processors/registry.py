"""
The registry for processors that contains framework-specific implementations of data processors.

Having a registry of all available classes is convenient for retrieving an instance based on a configuration at run-time.
"""
import logging
from collections import OrderedDict
from typing import Literal, Tuple

from plato.processors import (base, pipeline,
                              mistnet_torch_randomized_response,
                              mistnet_torch_unbatch,
                              mistnet_torch_send_ndarray_feature,
                              mistnet_torch_receive_ndarray_feature)

from plato.config import Config

registered_processors = OrderedDict([
    ('base', base.Processor),
    ('mistnet_torch_randomized_response',
     mistnet_torch_randomized_response.Processor),
    ('mistnet_torch_unbatch', mistnet_torch_unbatch.Processor),
    ('mistnet_torch_send_ndarray_feature',
     mistnet_torch_send_ndarray_feature.Processor),
    ('mistnet_torch_receive_ndarray_feature',
     mistnet_torch_receive_ndarray_feature.Processor),
])


def get(user: Literal["client", "server"], *args,
        **kwargs) -> Tuple[pipeline.Processor, pipeline.Processor]:
    """Get an instance of the processor."""

    send_processors = []
    receive_processors = []
    if user == "server":
        config = Config().server
    elif user == "client":
        config = Config().clients
    else:
        config = {}

    if hasattr(config, 'send_processors') and isinstance(
            config.send_processors, list):
        send_processors = config.send_processors
    if hasattr(config, 'receive_processors') and isinstance(
            config.receive_processors, list):
        receive_processors = config.receive_processors

    for processor in send_processors:
        logging.info("%s: Using Processor for sending payload: %s", user,
                     processor)
    for processor in receive_processors:
        logging.info("%s: Using Processor for receiving payload: %s", user,
                     processor)

    send_processors = list(
        map(lambda name: registered_processors[name](*args, **kwargs),
            send_processors))
    receive_processors = list(
        map(lambda name: registered_processors[name](*args, **kwargs),
            receive_processors))

    return pipeline.Processor(send_processors), pipeline.Processor(
        receive_processors)
