"""
A cross-silo federated learning server using federated averaging, as either edge or central servers.
"""

import asyncio
import logging
import os
import numpy as np

from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.processors import registry as processor_registry
from plato.samplers import registry as samplers_registry
from plato.samplers import all_inclusive
from plato.servers import fedavg
from plato.utils import csv_processor


class Server(fedavg.Server):
    """Cross-silo federated learning server using federated averaging."""

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

        self.current_global_round = 0
        self.average_accuracy = 0

        if Config().is_edge_server():
            # An edge client waits for the event that a certain number of
            # aggregations are completed
            self.model_aggregated = asyncio.Event()

            # An edge client waits for the event that a new global round begins
            # before starting the first round of local aggregation
            self.new_global_round_begins = asyncio.Event()

            edge_server_id = Config().args.id - Config().clients.total_clients

            # Compute the number of clients in each silo for edge servers
            edges_total_clients = [
                len(i)
                for i in np.array_split(
                    np.arange(Config().clients.total_clients),
                    Config().algorithm.total_silos,
                )
            ]
            self.total_clients = edges_total_clients[edge_server_id - 1]

            self.clients_per_round = [
                len(i)
                for i in np.array_split(
                    np.arange(Config().clients.per_round),
                    Config().algorithm.total_silos,
                )
            ][edge_server_id - 1]

            if hasattr(Config().trainer, "max_concurrency"):
                launched_total_clients = min(
                    Config().trainer.max_concurrency
                    * max(1, Config().gpu_count())
                    * Config().algorithm.total_silos,
                    Config().clients.per_round,
                )
            else:
                launched_total_clients = Config().clients.per_round

            edges_launched_clients = [
                len(i)
                for i in np.array_split(
                    np.arange(launched_total_clients), Config().algorithm.total_silos
                )
            ]
            starting_client_id = sum(edges_launched_clients[: edge_server_id - 1])
            launched_clients = edges_launched_clients[edge_server_id - 1]
            self.launched_clients = list(
                range(starting_client_id + 1, starting_client_id + 1 + launched_clients)
            )

            starting_client_id = sum(edges_total_clients[: edge_server_id - 1])
            self.clients_pool = list(
                range(
                    starting_client_id + 1, starting_client_id + 1 + self.total_clients
                )
            )

            logging.info(
                "[Edge server #%d (#%d)] Started training on %d clients with %d per round.",
                Config().args.id,
                os.getpid(),
                self.total_clients,
                self.clients_per_round,
            )

            self.recorded_items = ["global_round"] + self.recorded_items

        # Compute the number of clients for the central server
        if Config().is_central_server():
            self.clients_per_round = Config().algorithm.total_silos
            self.total_clients = self.clients_per_round

            logging.info(
                "The central server starts training with %s edge servers.",
                self.total_clients,
            )

    def configure(self):
        """
        Booting the federated learning server by setting up the data, model, and
        creating the clients.
        """
        if Config().is_edge_server():
            logging.info(
                "Configuring edge server #%d as a %s server.",
                Config().args.id,
                Config().algorithm.type,
            )
            logging.info(
                "[Edge server #%d (#%d)] Training with %s local aggregation rounds.",
                Config().args.id,
                os.getpid(),
                Config().algorithm.local_rounds,
            )

            self.init_trainer()
            self.trainer.set_client_id(Config().args.id)

            # Prepares this server for processors that processes outbound and inbound
            # data payloads
            self.outbound_processor, self.inbound_processor = processor_registry.get(
                "Server", server_id=os.getpid(), trainer=self.trainer
            )

            if (
                hasattr(Config().server, "edge_do_test")
                and Config().server.edge_do_test
            ):
                self.datasource = datasources_registry.get(client_id=Config().args.id)
                self.testset = self.datasource.get_test_set()

                if hasattr(Config().data, "testset_sampler"):
                    # Set the sampler for test set
                    self.testset_sampler = samplers_registry.get(
                        self.datasource, Config().args.id, testing=True
                    )
                else:
                    if hasattr(Config().data, "testset_size"):
                        self.testset_sampler = all_inclusive.Sampler(
                            self.datasource, testing=True
                        )

            # Initialize path of the result .csv file
            result_path = Config().params["result_path"]
            result_csv_file = f"{result_path}/edge_{os.getpid()}.csv"
            csv_processor.initialize_csv(
                result_csv_file, self.recorded_items, result_path
            )
        else:
            super().configure()

    async def select_clients(self, for_next_batch=False):
        if Config().is_edge_server() and not for_next_batch:
            if self.current_round == 0:
                # Wait until this edge server is selected by the central server
                # to avoid the edge server selects clients and clients begin training
                # before the edge server is selected
                await self.new_global_round_begins.wait()
                self.new_global_round_begins.clear()

        await super().select_clients(for_next_batch=for_next_batch)

    def customize_server_response(self, server_response: dict) -> dict:
        """Wrap up generating the server response with any additional information."""
        if Config().is_central_server():
            server_response["current_global_round"] = self.current_round
        return server_response

    async def _process_reports(self):
        """Process the client reports by aggregating their weights."""
        # To pass the client_id == 0 assertion during aggregation
        self.trainer.set_client_id(0)
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

        if Config().is_edge_server():
            self.trainer.set_client_id(Config().args.id)

        # Testing the model accuracy
        if (Config().is_edge_server() and Config().clients.do_test) or (
            Config().is_central_server()
            and hasattr(Config().server, "edge_do_test")
            and Config().server.edge_do_test
        ):
            # Compute the average accuracy from client reports
            self.average_accuracy = self.accuracy_averaging(self.updates)
            logging.info(
                "[%s] Average client accuracy: %.2f%%.",
                self,
                100 * self.average_accuracy,
            )
        elif Config().is_central_server() and Config().clients.do_test:
            # Compute the average accuracy from client reports
            total_samples = sum(update.report.num_samples for update in self.updates)
            self.average_accuracy = (
                sum(
                    update.report.average_accuracy * update.report.num_samples
                    for update in self.updates
                )
                / total_samples
            )

            logging.info(
                "[%s] Average client accuracy: %.2f%%.",
                self,
                100 * self.average_accuracy,
            )

        if (
            Config().is_central_server()
            and hasattr(Config().server, "do_test")
            and Config().server.do_test
        ):
            # Test the updated model directly at the central server
            self.accuracy = self.trainer.test(self.testset, self.testset_sampler)
            if hasattr(Config().trainer, "target_perplexity"):
                logging.info(
                    "[%s] Global model perplexity: %.2f\n", self, self.accuracy
                )
            else:
                logging.info(
                    "[%s] Global model accuracy: %.2f%%\n", self, 100 * self.accuracy
                )
        elif (
            Config().is_edge_server()
            and hasattr(Config().server, "edge_do_test")
            and Config().server.edge_do_test
        ):
            # Test the aggregated model directly at the edge server
            self.accuracy = self.trainer.test(self.testset, self.testset_sampler)
            if hasattr(Config().trainer, "target_perplexity"):
                logging.info(
                    "[%s] Aggregated model perplexity: %.2f\n", self, self.accuracy
                )
            else:
                logging.info(
                    "[%s] Aggregated model accuracy: %.2f%%\n",
                    self,
                    100 * self.accuracy,
                )
        else:
            self.accuracy = self.average_accuracy

        await self.wrap_up_processing_reports()

    async def wrap_up_processing_reports(self):
        """Wrap up processing the reports with any additional work."""
        # Record results into a .csv file
        if Config().is_central_server():
            await super().wrap_up_processing_reports()

        if Config().is_edge_server():
            new_row = []
            for item in self.recorded_items:
                item_value = self.get_record_items_values()[item]
                new_row.append(item_value)

            result_csv_file = f"{Config().params['result_path']}/edge_{os.getpid()}.csv"
            csv_processor.write_csv(result_csv_file, new_row)

            if hasattr(Config().clients, "do_test") and Config().clients.do_test:
                # Updates the log for client test accuracies
                accuracy_csv_file = (
                    f"{Config().params['result_path']}/edge_{os.getpid()}_accuracy.csv"
                )

                for update in self.updates:
                    accuracy_row = [
                        self.current_round,
                        update.client_id,
                        update.report.accuracy,
                    ]
                    csv_processor.write_csv(accuracy_csv_file, accuracy_row)

            # When a certain number of aggregations are completed, an edge client
            # needs to be signaled to send a report to the central server
            if self.current_round == Config().algorithm.local_rounds:
                logging.info(
                    "[Server #%d] Completed %s rounds of local aggregation.",
                    os.getpid(),
                    Config().algorithm.local_rounds,
                )
                self.model_aggregated.set()

                self.current_round = 0
                self.current_global_round += 1

    def get_record_items_values(self):
        """Get values will be recorded in result csv file."""
        record_items_values = super().get_record_items_values()

        record_items_values["global_round"] = self.current_global_round
        record_items_values["average_accuracy"] = self.average_accuracy
        record_items_values["edge_agg_num"] = Config().algorithm.local_rounds
        record_items_values["local_epoch_num"] = Config().trainer.epochs

        return record_items_values

    async def wrap_up(self):
        """Wrapping up when each round of training is done."""
        if Config().is_central_server():
            await super().wrap_up()
