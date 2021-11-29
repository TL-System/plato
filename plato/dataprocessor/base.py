"""
DataProcessor for pre-processing payload before sending or after receiving.
"""
from collections.abc import Iterable
from typing import Any


class DataProcessor:
    """
    DataProcessor class.
    Base DataProcessor class implementation do nothing on the data.
    """
    def __init__(self, *args, **kwargs) -> None:
        """Constructor for DataProcessor"""
        pass

    def process(self, data: Any) -> Any:
        """
        Data processing implementation.
        Implement this method while inheriting the class.
        """
        return data

    def process_iterable(self, data: Iterable) -> Iterable:
        """
        Data processing for an Iterable of data.
        """
        return map(lambda d: self.process(d), data)
