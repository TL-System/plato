"""
A federated learning client for FEI.
"""
import logging
import math
from dataclasses import dataclass

from afl import afl_client
from plato.clients import simple
from plato.config import Config


@dataclass
class Report(afl_client.Report):
    """A client report containing the valuation, to be sent to the FEI federated learning server."""


class Client(afl_client.Client):
    """A federated learning client for FEI."""
    
