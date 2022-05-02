"""
A federated learning server with RL Agent.
"""
import asyncio
import logging
import os
from abc import abstractmethod

from plato.algorithms import registry as algorithms_registry
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.models import registry as models_registry
from plato.processors import registry as processor_registry
from plato.servers import fedavg
from plato.trainers import registry as trainers_registry


class RLServer(fedavg.Server):
    """ A federated learning server with an RL Agent. """
    def __init__(self, agent, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        self.agent = agent

    def configure(self):
        """ Booting the federated learning server by setting up
        the data, model, and creating the clients.
            Called every time when reseting a new RL episode.
        """
        super().configure()
        logging.info("[Server #%d] Configuring the server for episode %d",
                     os.getpid(), self.agent.current_episode)

        self.current_round = 0

        self.load_trainer()

        # Prepares this server for processors that processes outbound and inbound
        # data payloads
        self.outbound_processor, self.inbound_processor = processor_registry.get(
            "Server", server_id=os.getpid(), trainer=self.trainer)

        if not Config().clients.do_test:
            dataset = datasources_registry.get(client_id=0)
            self.testset = dataset.get_test_set()

    def load_trainer(self):
        """ Setting up the global model to be trained via federated learning. """
        if self.trainer is None and self.custom_trainer is None:
            self.trainer = trainers_registry.get(model=self.model)
        elif self.custom_trainer is not None:
            self.trainer = self.custom_trainer()
            self.custom_trainer = None

        self.trainer.set_client_id(0)

        # Reset model for new episode
        self.trainer.model = models_registry.get()

        self.algorithm = algorithms_registry.get(self.trainer)

    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using smart weighting."""
        # Extract weights udpates from the client updates
        weights_received = self.extract_client_updates(updates)
        self.update_state()

        # Extract the total number of samples
        num_samples = [report.num_samples for (report, __, __) in updates]
        self.total_samples = sum(num_samples)

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        # e.g., wait for the new action from RL agent
        # if the action affects the global aggregation
        self.agent.num_samples = num_samples
        await self.agent.prep_agent_update()
        await self.update_action()

        # Use adaptive weighted average
        for i, update in enumerate(weights_received):
            for name, delta in update.items():
                avg_update[name] += delta * self.smart_weighting[i][0]

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update

    async def customize_server_response(self, server_response):
        """ Wrap up generating the server response with any additional information. """
        server_response['current_round'] = self.current_round
        return server_response

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

    async def wrap_up(self):
        """ Wrapping up when each round of training is done. """
        self.save_to_checkpoint()

        if self.agent.reset_env:
            self.agent.reset_env = False
            self.configure()
        if self.agent.finished:
            await self.close()

    @abstractmethod
    def prep_state(self):
        """ Wrap up the state update to RL Agent. """
        return

    @abstractmethod
    def apply_action(self):
        """ Apply action update from RL Agent to FL Env. """
