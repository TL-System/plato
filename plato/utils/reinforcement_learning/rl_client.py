"""
RL on clients
"""
import asyncio
import logging
from abc import abstractmethod
import time

from plato.clients import base
from plato.algorithms import registry as algorithms_registry
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.processors import registry as processor_registry
from plato.samplers import registry as samplers_registry
from plato.trainers import registry as trainers_registry
from plato.utils.reinforcement_learning.policies import td3
from plato.clients import simple

class Report(simple.Report):
    average_accuracy: float
    client_id: str

class RLClient(simple.Client):
    """A federated learning client that uses TD3 to learn"""
    
    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model=model,
                         datasource=datasource,
                         algorithm=algorithm,
                         trainer=trainer)
   