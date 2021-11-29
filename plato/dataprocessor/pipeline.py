"""
DataPipeline for passing data through all DataProcessors
"""
from typing import Iterable, Literal

from plato.dataprocessor import registry


class DataPipeline:
    """
    DataPipeline class
    Pipelining a list of DataProcessors from config
    """
    def __init__(self, user: Literal["server", "client"], *args,
                 **kwargs) -> None:
        """Constructor for DataPipeline"""
        self.processors = registry.get(user, *args, **kwargs)

    def process(self, data):
        """
        Data pipelining implementation.
        """
        for processor in self.processors:
            data = processor(data)
        return data

    def process_iterable(self, data: Iterable):
        """
        Data processing for an Iterable of data.
        """
        return map(lambda d: self.process(d), data)
