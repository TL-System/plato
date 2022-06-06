"""
The Processor class is designed for pre-processing data payloads before or after they
are transmitted over the network between the clients and the servers.
"""
from abc import abstractmethod
from collections.abc import Iterable
from typing import Any


class Processor:
    """
    The base Processor class does nothing on the data payload.
    """
    def __init__(self, trainer=None, **kwargs) -> None:
        """ Constructor for Processor. """
        self.trainer = trainer

    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Processing a data payload.
        """
        return data

    def process_iterable(self, data: Iterable) -> Iterable:
        """
        Processing an Iterable of data payload.
        """
        return map(self.process, data)
