"""
Implementation of the APFL's clients.

"""

import logging

from plato.clients import pers_simple
from plato.config import Config


class Client(pers_simple.Client):
    """A personalized federated learning client with APFL method."""
