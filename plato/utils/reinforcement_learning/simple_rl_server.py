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
from plato.utils import csv_processor


class RLServer(fedavg.Server):
    """ A federated learning server with RL Agent. """
    def __init__(self, agent, trainer=None):
        super().__init__(trainer=trainer)
        self.agent = agent

    async def update_action(self):
        if self.agent.current_step == 0:
            logging.info("[RL Agent] Preparing initial action...")
            self.agent.prep_action()
        else:
            await self.agent.action_updated.wait()
            self.agent.action_updated.clear()

        self.apply_action()

    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using smart weighting."""
        # Extract weights udpates from the client updates
        weights_received = self.extract_client_updates(updates)

        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (report, __, __) in updates])

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        # e.g., wait for the new action from RL agent
        # if the action affects the global aggregation
        await self.update_action()

        # Use adaptive weighted average
        for i, update in enumerate(weights_received):
            for name, delta in update.items():
                avg_update[name] += delta * self.smart_weighting[i]

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update

    def update_state(self):
        """ Wrap up the state update to RL Agent. """
        # Pass new state to RL Agent
        self.agent.new_state = self.prep_state()
        self.agent.process_env_update()

    @abstractmethod
    def prep_state(self):
        """ Wrap up the state update to RL Agent. """
        return

    @abstractmethod
    def apply_action(self):
        """ Apply action update from RL Agent to FL Env. """

    async def wrap_up(self):
        """ Wrapping up when each round of training is done. """
        self.save_to_checkpoint()

        self.update_state()
        await self.agent.prep_agent_update()
        if self.agent.reset_env:
            self.agent.reset_env = False
            self.configure()
            await self.wrap_up()
        if self.agent.finished:
            await self.close()

    def configure(self):
        """ Booting the federated learning server by setting up
        the data, model, and creating the clients. 
        
            Called every time when reseting a new RL episode.
        """
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

        # Initialize the csv file which will record results
        if self.agent.current_episode == 1 and hasattr(Config(), 'results'):
            results_dir = Config().results_dir
            result_csv_file = f'{results_dir}/{os.getpid()}_episode_result.csv'
            csv_processor.initialize_csv(result_csv_file, self.recorded_items,
                                         results_dir)

    def load_trainer(self):
        """ Setting up the global model to be trained via federated learning. """
        if self.trainer is None:
            self.trainer = trainers_registry.get(model=self.model)

        self.trainer.set_client_id(0)

        # Reset model for new episode
        self.trainer.model = models_registry.get()

        self.algorithm = algorithms_registry.get(self.trainer)
