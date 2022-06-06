"""
Implements a pipeline of processors for data payloads to pass through.
"""
from typing import Any, List

from plato.processors import base


class Processor(base.Processor):
    """
    Pipelining a list of Processors from the configuration file.
    """
    def __init__(self, processors: List[base.Processor], *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.processors = processors

    def process(self, data: Any) -> Any:
        """
        Implementing a pipeline of Processors for data payloads.
        """
        for processor in self.processors:
            data = processor.process(data)

        return data
