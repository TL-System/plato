"""Base classes for MindSpore models."""
from abc import abstractstaticmethod
import mindspore.nn as nn


class Model(nn.Cell):
    """The base class for by all the MindSpore models."""
    @abstractstaticmethod
    def is_valid_model_type(model_type: str) -> bool:
        """Is the model name string a valid name for models in this class?"""

    @abstractstaticmethod
    def get_model_from_type(model_type: str) -> 'Model':
        """Returns an instance of this class as described by the model_type string."""
