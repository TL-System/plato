"""
DataPipeline for passing data through all Processors
"""
from typing import Any, List

from plato.processors import base


class Processor(base.Processor):
    """
    DataPipeline class
    Pipelining a list of Processors from config
    """
    def __init__(self, processors: List[base.Processor], *args,
                 **kwargs) -> None:
        """Constructor for DataPipeline"""
        self.processors = processors

    def process(self, data: Any) -> Any:
        """
        Data pipelining implementation.
        """
        for processor in self.processors:
            data = processor.process(data)
        return data
