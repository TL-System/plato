"""
A federated learning client with support for Adaptive Synchronization Frequency.
"""

import logging
import random
import time
from dataclasses import dataclass

from models import registry as models_registry
from datasets import registry as datasets_registry
from trainers import registry as trainers_registry
from dividers import iid, biased, sharded
from utils import dists
from config import Config
from clients import SimpleClient


class AdaptiveSyncClient(SimpleClient):
    """A federated learning client with support for Adaptive Synchronization
    Frequency.
    """
    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        if 'sync_frequency' in server_response:
            Config().trainer = Config().trainer._replace(
                epochs=server_response['sync_frequency'])
