"""
DataPipeline for passing data through all DataProcessors
"""
from typing import Any, List

from plato.dataprocessor import base


class DataProcessor(base.DataProcessor):
    """
    DataPipeline class
    Pipelining a list of DataProcessors from config
    """
    def __init__(self, dataprocessors: List[base.DataProcessor]) -> None:
        """Constructor for DataPipeline"""
        self.processors = dataprocessors

    def process(self, data: Any) -> Any:
        """
        Data pipelining implementation.
        """
        for processor in self.processors:
            data = processor.process(data)
        return data
