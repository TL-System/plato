"""
Base class for datasets.
"""

from abc import ABC, abstractstaticmethod


class Dataset(ABC):
    """
    The training or testing dataset that accommodates custom augmentation and transforms.
    """
    def __init__(self, path):
        self._path = path

    @abstractstaticmethod
    def num_train_examples() -> int:
        pass

    @abstractstaticmethod
    def num_test_examples() -> int:
        pass

    @abstractstaticmethod
    def num_classes() -> int:
        pass

    @abstractstaticmethod
    def get_train_set() -> 'Dataset':
        pass

    @abstractstaticmethod
    def get_test_set() -> 'Dataset':
        pass
