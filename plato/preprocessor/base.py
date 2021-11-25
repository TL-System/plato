"""The base class of preprocessors"""

from abc import abstractmethod


class Preprocessor:
    """Base preprocessor class."""
    def __init__(self) -> None:
        pass

    @abstractmethod
    def process(self, data):
        """Process a block of data, return a block of data."""

    def stream_process(self, iterator):
        """"Process a stream of data from an iterator, should return via yield
        or return an iterator"""
        for data in iterator:
            yield self.process(data)
