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
from plato.utils.reinforcement_learning import rl_agent


class Report(simple.Report):
    average_accuracy: float
    client_id: str

class RLClient(simple.Client):
    """A federated learning client that uses RL methods to learn"""
    
    #maybe only two functions?? *look into
    def __init__(self, agent, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        self.agent = agent

    def reset(self):
        """ Resetting the model, trainer, and algorithm on the client. """
        logging.info("Reconfiguring the client for episode %d",
                     self.agent.current_episode)

        self.model = None
        self.trainer = None
        self.algorithm = None

        self.current_round = 0

    async def customize_client_response(self, client_response):
        """ Wrap up generating the client response with any additional information. """
        client_response['current_round'] = self.current_round
        return client_response

    async def update_action(self):
        """ Updating the RL agent's actions. """
        if self.agent.current_step == 0:
            logging.info("[RL Agent] Preparing initial action...")
            self.agent.prep_action()
        else:
            await self.agent.action_updated.wait()
            self.agent.action_updated.clear()

        self.apply_action()

    def update_state(self):
        """ Wrap up the state update to RL Agent. """
        # Pass new state to RL Agent
        self.agent.new_state = self.prep_state()
        self.agent.process_env_update()

    @abstractmethod
    def train(self):
        """Machine Learning algorithm on client"""

    @abstractmethod
    def calc_loss(self):
        """Calculates the loss"""

    @abstractmethod
    def prep_state(self):
        """ Wrap up the state update to RL Agent. """

    @abstractmethod
    def apply_action(self):
        """ Apply action update from RL Agent to FL Env. """


        

        
   