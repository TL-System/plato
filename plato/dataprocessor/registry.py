"""
The registry for dataprocessors that contains framework-specific implementations of data processors.

Having a registry of all available classes is convenient for retrieving an instance based on a configuration at run-time.
"""
import logging
from collections import OrderedDict
from typing import Literal, Tuple

from plato.dataprocessor import (base, pipeline)

from plato.config import Config

registered_dataprocessors = OrderedDict([
    ('base', base.DataProcessor),
])


def get(user: Literal["client", "server"], *args,
        **kwargs) -> Tuple[pipeline.DataProcessor, pipeline.DataProcessor]:
    """Get an instance of the dataprocessor."""

    send_dataprocessors = []
    receive_dataprocessors = []
    if user == "server":
        config = Config().server
    elif user == "client":
        config = Config().clients
    else:
        config = {}

    if hasattr(config, 'send_dataprocessors') and isinstance(
            config.send_dataprocessors, list):
        send_dataprocessors = config.send_dataprocessors
    if hasattr(config, 'receive_dataprocessors') and isinstance(
            config.receive_dataprocessors, list):
        receive_dataprocessors = config.receive_dataprocessors

    for processor in send_dataprocessors:
        logging.info("%s: Using DataProcessor for sending payload: %s", user,
                     processor)
    for processor in receive_dataprocessors:
        logging.info("%s: Using DataProcessor for receiving payload: %s", user,
                     processor)

    send_dataprocessors = list(
        map(lambda name: registered_dataprocessors[name](*args, **kwargs),
            send_dataprocessors))
    receive_dataprocessors = list(
        map(lambda name: registered_dataprocessors[name](*args, **kwargs),
            receive_dataprocessors))

    return pipeline.DataProcessor(send_dataprocessors), pipeline.DataProcessor(
        receive_dataprocessors)
