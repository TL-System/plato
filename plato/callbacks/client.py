"""
Defines the TrainerCallback class, which is the abstract base class to be subclassed
when creating new client callbacks.

Defines a default callback to print local training progress.
"""

from abc import ABC


class ClientCallback(ABC):
    """
    The abstract base class to be subclassed when creating new client callbacks.
    """


class PrintProgressCallback(ClientCallback):
    """
    A callback which prints a message when needed.
    """
