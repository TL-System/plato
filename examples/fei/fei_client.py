"""
A federated learning client for FEI.
"""
from dataclasses import dataclass

from afl import afl_client


@dataclass
class Report(afl_client.Report):
    """A client report containing the valuation, to be sent to the FEI federated learning server."""


class Client(afl_client.Client):
    """A federated learning client for FEI."""
