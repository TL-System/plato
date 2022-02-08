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
from plato.servers import base
from plato.trainers import registry as trainers_registry
from plato.utils import csv_processor


class RLServer(base.Server):
    """ A federated learning server with RL Agent. """

    def __init__(self, agent, model=None, algorithm=None, trainer=None):
        super().__init__()
        self.agent = agent

        self.model = model
        self.algorithm = algorithm
        self.trainer = trainer

        self.testset = None
        self.total_samples = 0

        self.total_clients = Config().clients.total_clients
        self.clients_per_round = Config().clients.per_round

        logging.info(
            "[Server #%d] Started training on %s clients with %s per round.",
            os.getpid(), self.total_clients, self.clients_per_round)

        if hasattr(Config(), 'results'):
            recorded_items = Config().results.types
            self.recorded_items = [
                x.strip() for x in recorded_items.split(',')
            ]

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
        if self.agent.current_episode == 0 and hasattr(Config(), 'results'):
            result_dir = Config().result_dir
            result_csv_file = f'{result_dir}/{os.getpid()}_episode_result.csv'
            csv_processor.initialize_csv(result_csv_file, self.recorded_items,
                                         result_dir)

    def load_trainer(self):
        """ Setting up the global model to be trained via federated learning. """
        if self.trainer is None:
            self.trainer = trainers_registry.get(model=self.model)

        self.trainer.set_client_id(0)

        # Reset model for new episode
        self.trainer.model = models_registry.get()

        if self.algorithm is None:
            self.algorithm = algorithms_registry.get(self.trainer)

    def extract_client_updates(self, updates):
        """Extract the model weight updates from client updates."""
        weights_received = [payload for (__, payload, __) in updates]
        return self.algorithm.compute_weight_updates(weights_received)

    async def aggregate_weights(self, updates):
        """Aggregate the reported weight updates from the selected clients."""
        update = await self.federated_averaging(updates)
        updated_weights = self.algorithm.update_weights(update)
        self.algorithm.load_weights(updated_weights)

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

    async def process_reports(self):
        """Process the client reports by aggregating their weights."""
        await self.aggregate_weights(self.updates)

        # Testing the global model accuracy
        if Config().clients.do_test:
            # Compute the average accuracy from client reports
            self.accuracy = self.accuracy_averaging(self.updates)
            logging.info(
                '[Server #{:d}] Average client accuracy: {:.2f}%.'.format(
                    os.getpid(), 100 * self.accuracy))
        else:
            # Testing the updated model directly at the server
            self.accuracy = await self.trainer.server_test(self.testset)

            logging.info(
                '[Server #{:d}] Global model accuracy: {:.2f}%\n'.format(
                    os.getpid(), 100 * self.accuracy))

        await self.wrap_up_processing_reports()

    async def wrap_up_processing_reports(self):
        """ Wrap up processing the reports with any additional work. """
        if hasattr(Config(), 'results'):
            new_row = []

            for item in self.recorded_items:
                item_value = {
                    'round':
                    self.current_round,
                    'accuracy':
                    self.accuracy * 100,
                    'elapsed_time':
                    self.wall_time - self.initial_wall_time,
                    'round_time':
                    max([
                        report.training_time
                        for (report, __, __) in self.updates
                    ]),
                }[item]
                new_row.append(item_value)

            result_csv_file = f'{Config().result_dir}/{os.getpid()}.csv'
            csv_processor.write_csv(result_csv_file, new_row)

    @staticmethod
    def accuracy_averaging(updates):
        """Compute the average accuracy across clients."""
        # Get total number of samples
        total_samples = sum(
            [report.num_samples for (report, __, __) in updates])

        # Perform weighted averaging
        accuracy = 0
        for (report, __, __) in updates:
            accuracy += report.accuracy * (report.num_samples / total_samples)

        return accuracy

    def customize_server_payload(self, payload):
        """ Customize the server payload before sending to the client. """
        return payload

    async def update_action(self):
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

        self.update_state()
        await self.agent.prep_agent_update()
        if self.agent.reset_env:
            self.agent.reset_env = False
            self.configure()
            await self.wrap_up()
        if self.agent.finished:
            await self.close()

    @abstractmethod
    def prep_state(self):
        """ Wrap up the state update to RL Agent. """
        return

    @abstractmethod
    def apply_action(self):
        """ Apply action update from RL Agent to FL Env. """
