"""The base class of reversable preprocessors"""

from abc import abstractmethod
from plato.preprocessor import base


class Preprocessor(base.Preprocessor):
    """Base reversable preprocessor class."""
    def __init__(self):
        """Base reversable preprocessor constructor."""
        super().__init__()

    @abstractmethod
    def unprocess(self, data):
        """Reverse the processing of a processed block of data, return a block
        of data."""

    def stream_unprocess(self, iterator):
        """"Reverse the processing of a stream of data from an iterator, should
        return via yield or return an iterator"""
        for data in iterator:
            yield self.unprocess(data)
