"""
The registry for dataprocessors that contains framework-specific implementations of data processors.

Having a registry of all available classes is convenient for retrieving an instance based on a configuration at run-time.
"""
import logging
from collections import OrderedDict
from typing import Literal

from plato.dataprocessor import (
    base, )

from plato.config import Config

registered_dataprocessors = OrderedDict([
    ('base', base.DataProcessor),
])


def get(user: Literal["client", "server"], *args,
        **kwargs) -> list[base.DataProcessor]:
    """Get an instance of the dataprocessor."""

    dataprocessors = []
    if user == "server":
        if hasattr(Config().server, 'dataprocessors') and isinstance(
                Config().server.dataprocessors, list):
            dataprocessors = Config().server.dataprocessors
    elif user == "client":
        if hasattr(Config().clients, 'dataprocessors') and isinstance(
                Config().clients.dataprocessors, list):
            dataprocessors = Config().server.dataprocessors

    for dataprocessor in dataprocessors:
        logging.info("%s: Using DataProcessor: %s", user, dataprocessor)

    return map(lambda name: registered_dataprocessors[name](*args, **kwargs),
               dataprocessors)
