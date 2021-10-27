"""
A personalized federated learning client.
"""
import logging
import pickle
import sys
from dataclasses import dataclass

from plato.clients import simple


@dataclass
class Report(simple.Report):
    """A client report."""


class Client(simple.Client):
    """A federated learning client."""
    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model=model,
                         datasource=datasource,
                         algorithm=algorithm,
                         trainer=trainer)
        pass