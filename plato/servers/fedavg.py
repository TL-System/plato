"""
A simple federated learning server using federated averaging.
"""

import asyncio
import logging
import os

from plato.algorithms import registry as algorithms_registry
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.processors import registry as processor_registry
from plato.servers import base
from plato.trainers import registry as trainers_registry
from plato.utils import csv_processor
from plato.samplers import all_inclusive


class Server(base.Server):
    """Federated learning server using federated averaging."""

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__()

        self.custom_model = model
        self.model = None

        self.custom_algorithm = algorithm
        self.algorithm = None

        self.custom_trainer = trainer
        self.trainer = None

        self.custom_datasource = datasource
        self.datasource = None

        self.testset = None
        self.testset_sampler = None
        self.total_samples = 0

        self.total_clients = Config().clients.total_clients
        self.clients_per_round = Config().clients.per_round

        logging.info(
            "[Server #%d] Started training on %d clients with %d per round.",
            os.getpid(),
            self.total_clients,
            self.clients_per_round,
        )

        recorded_items = Config().params["result_types"]
        self.recorded_items = [x.strip() for x in recorded_items.split(",")]

    def configure(self):
        """
        Booting the federated learning server by setting up the data, model, and
        creating the clients.
        """
        super().configure()

        total_rounds = Config().trainer.rounds
        target_accuracy = None
        target_perplexity = None

        if hasattr(Config().trainer, "target_accuracy"):
            target_accuracy = Config().trainer.target_accuracy
        elif hasattr(Config().trainer, "target_perplexity"):
            target_perplexity = Config().trainer.target_perplexity

        if target_accuracy:
            logging.info(
                "Training: %s rounds or accuracy above %.1f%%\n",
                total_rounds,
                100 * target_accuracy,
            )
        elif target_perplexity:
            logging.info(
                "Training: %s rounds or perplexity below %.1f\n",
                total_rounds,
                target_perplexity,
            )
        else:
            logging.info("Training: %s rounds\n", total_rounds)

        self.init_trainer()

        # Prepares this server for processors that processes outbound and inbound
        # data payloads
        self.outbound_processor, self.inbound_processor = processor_registry.get(
            "Server", server_id=os.getpid(), trainer=self.trainer
        )

        if not (hasattr(Config().server, "do_test") and not Config().server.do_test):
            if self.datasource is None and self.custom_datasource is None:
                self.datasource = datasources_registry.get(client_id=0)
            elif self.datasource is None and self.custom_datasource is not None:
                self.datasource = self.custom_datasource()

            self.testset = self.datasource.get_test_set()
            if hasattr(Config().data, "testset_size"):
                self.testset_sampler = all_inclusive.Sampler(
                    self.datasource, testing=True
                )

        # Initialize the csv file which will record results
        result_csv_file = f"{Config().params['result_path']}/{os.getpid()}.csv"
        csv_processor.initialize_csv(
            result_csv_file, self.recorded_items, Config().params["result_path"]
        )

        # Initialize the test accuracy csv file if clients compute locally
        if hasattr(Config().clients, "do_test") and Config().clients.do_test:
            accuracy_csv_file = (
                f"{Config().params['result_path']}/{os.getpid()}_accuracy.csv"
            )
            accuracy_headers = ["round", "client_id", "accuracy"]
            csv_processor.initialize_csv(
                accuracy_csv_file, accuracy_headers, Config().params["result_path"]
            )

    def init_trainer(self):
        """Setting up the global model, trainer, and algorithm."""
        if self.model is None and self.custom_model is not None:
            self.model = self.custom_model

        if self.trainer is None and self.custom_trainer is None:
            self.trainer = trainers_registry.get(model=self.model)
        elif self.trainer is None and self.custom_trainer is not None:
            self.trainer = self.custom_trainer(model=self.model)

        if self.algorithm is None and self.custom_algorithm is None:
            self.algorithm = algorithms_registry.get(trainer=self.trainer)
        elif self.algorithm is None and self.custom_algorithm is not None:
            self.algorithm = self.custom_algorithm(trainer=self.trainer)

    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using federated averaging."""
        # Extract the total number of samples
        self.total_samples = sum(update.report.num_samples for update in updates)

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        for i, update in enumerate(deltas_received):
            report = updates[i].report
            num_samples = report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples)

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update

    async def _process_reports(self):
        """Process the client reports by aggregating their weights."""
        weights_received = [update.payload for update in self.updates]

        weights_received = self.weights_received(weights_received)
        self.callback_handler.call_event("on_weights_received", self, weights_received)

        # Extract the current model weights as the baseline
        baseline_weights = self.algorithm.extract_weights()

        if hasattr(self, "aggregate_weights"):
            # Runs a server aggregation algorithm using weights rather than deltas
            logging.info(
                "[Server #%d] Aggregating model weights directly rather than weight deltas.",
                os.getpid(),
            )
            updated_weights = self.aggregate_weights(
                self.updates, baseline_weights, weights_received
            )

            # Loads the new model weights
            self.algorithm.load_weights(updated_weights)
        else:
            # Computes the weight deltas by comparing the weights received with
            # the current global model weights
            deltas_received = self.algorithm.compute_weight_deltas(
                baseline_weights, weights_received
            )

            # Runs a framework-agnostic server aggregation algorithm, such as
            # the federated averaging algorithm
            logging.info("[Server #%d] Aggregating model weight deltas.", os.getpid())
            deltas = await self.aggregate_deltas(self.updates, deltas_received)

            # Updates the existing model weights from the provided deltas
            updated_weights = self.algorithm.update_weights(deltas)

            # Loads the new model weights
            self.algorithm.load_weights(updated_weights)

        # The model weights have already been aggregated, now calls the
        # corresponding hook and callback
        self.weights_aggregated(self.updates)
        self.callback_handler.call_event("on_weights_aggregated", self, self.updates)

        # Testing the global model accuracy
        if hasattr(Config().server, "do_test") and not Config().server.do_test:
            # Compute the average accuracy from client reports
            self.accuracy = self.accuracy_averaging(self.updates)
            logging.info(
                "[%s] Average client accuracy: %.2f%%.", self, 100 * self.accuracy
            )
        else:
            # Testing the updated model directly at the server

            self.accuracy = self.trainer.test(self.testset, self.testset_sampler)

        if hasattr(Config().trainer, "target_perplexity"):
            logging.info("[%s] Global model perplexity: %.2f\n", self, self.accuracy)
        else:
            logging.info(
                "[%s] Global model accuracy: %.2f%%\n", self, 100 * self.accuracy
            )

        await self.wrap_up_processing_reports()

    async def wrap_up_processing_reports(self):
        """Wrap up processing the reports with any additional work."""
        # Record results into a .csv file
        new_row = []
        for item in self.recorded_items:
            item_value = self.get_record_items_values()[item]
            new_row.append(item_value)

        result_csv_file = f"{Config().params['result_path']}/{os.getpid()}.csv"
        csv_processor.write_csv(result_csv_file, new_row)

        if hasattr(Config().clients, "do_test") and Config().clients.do_test:
            # Updates the log for client test accuracies
            accuracy_csv_file = (
                f"{Config().params['result_path']}/{os.getpid()}_accuracy.csv"
            )

            for update in self.updates:
                accuracy_row = [
                    self.current_round,
                    update.client_id,
                    update.report.accuracy,
                ]
                csv_processor.write_csv(accuracy_csv_file, accuracy_row)

    def get_record_items_values(self):
        """Get values will be recorded in result csv file."""
        return {
            "round": self.current_round,
            "accuracy": self.accuracy,
            "elapsed_time": self.wall_time - self.initial_wall_time,
            "comm_time": max(update.report.comm_time for update in self.updates),
            "round_time": max(
                update.report.training_time + update.report.comm_time
                for update in self.updates
            ),
            "comm_overhead": self.comm_overhead,
        }

    @staticmethod
    def accuracy_averaging(updates):
        """Compute the average accuracy across clients."""
        # Get total number of samples
        total_samples = sum(update.report.num_samples for update in updates)

        # Perform weighted averaging
        accuracy = 0
        for update in updates:
            accuracy += update.report.accuracy * (
                update.report.num_samples / total_samples
            )

        return accuracy

    def weights_received(self, weights_received):
        """
        Event called after the updated weights have been received.
        """
        return weights_received

    def weights_aggregated(self, updates):
        """
        Method called after the updated weights have been aggregated.
        """
