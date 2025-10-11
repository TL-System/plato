"""
A cross-silo personalized federated learning server using MAML algorithm,
as either edge or central servers.
"""

import asyncio
import logging
import os

from plato.config import Config
from plato.servers import fedavg_cs
from plato.utils import csv_processor


class Server(fedavg_cs.Server):
    """Cross-silo federated learning server using federated averaging."""

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

        self.do_personalization_test = False

        # A list to store accuracy of clients' personalized models
        self.personalization_test_updates = []
        self.personalization_accuracy = 0

        self.round_time = 0
        self.comm_time = 0

        if Config().is_edge_server():
            # An edge client waits for the event that a certain number of clients
            # compute accuracy of their personalized models
            self.per_accuracy_aggregated = asyncio.Event()

    async def select_testing_clients(self):
        """Select a subset of the clients to test personalization."""
        self.do_personalization_test = True
        logging.info("\n[%s] Starting testing personalization.", self)

        self.current_round -= 1
        await super()._select_clients()

        if len(self.selected_clients) > 0:
            logging.info(
                "[%s] Sent the current meta model to %d clients for personalization test.",
                self,
                len(self.selected_clients),
            )

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        """Wrap up generating the server response with any additional information."""
        if self.do_personalization_test:
            server_response["personalization_test"] = True
        return server_response

    async def _process_reports(self):
        """Process the client reports by aggregating their weights."""
        if self.do_personalization_test:
            self.compute_personalization_accuracy()
            self.clients_processed()
        else:
            await super()._process_reports()

    def compute_personalization_accuracy(self):
        """ "Average accuracy of clients' personalized models."""
        accuracy = 0
        for report in self.personalization_test_updates:
            accuracy += report
        self.personalization_accuracy = accuracy / len(
            self.personalization_test_updates
        )

    def get_logged_items(self) -> dict:
        """Get items to be logged by the LogProgressCallback class in a .csv file."""
        return {
            "global_round": self.current_global_round,
            "round": self.current_round,
            "accuracy": self.accuracy * 100,
            "average_accuracy": self.average_accuracy * 100,
            "edge_agg_num": Config().algorithm.local_rounds,
            "local_epoch_num": Config().trainer.epochs,
            "elapsed_time": self.wall_time - self.initial_wall_time,
            "comm_time": self.comm_time,
            "round_time": self.round_time,
            "personalization_accuracy": self.personalization_accuracy * 100,
        }

    def clients_processed(self):
        """Additional work to be performed after client reports have been processed."""
        if self.do_personalization_test or Config().is_edge_server():
            # Record results
            new_row = []
            for item in self.recorded_items:
                item_value = self.get_logged_items()[item]
                new_row.append(item_value)

            if Config().is_edge_server():
                result_csv_file = (
                    f"{Config().params['result_path']}/edge_{os.getpid()}.csv"
                )
            else:
                result_csv_file = f"{Config().params['result_path']}/{os.getpid()}.csv"

            csv_processor.write_csv(result_csv_file, new_row)

            if Config().is_edge_server():
                if self.do_personalization_test:
                    self.per_accuracy_aggregated.set()
                    self.current_global_round += 1
                else:
                    # When a certain number of aggregations are completed, an edge client
                    # needs to be signaled to send a report to the central server
                    if self.current_round == Config().algorithm.local_rounds:
                        logging.info(
                            "[%s] Completed %s rounds of local aggregation.",
                            self,
                            Config().algorithm.local_rounds,
                        )
                        self.model_aggregated.set()

                        self.current_round = 0

        if not self.do_personalization_test:
            self.round_time = max(
                update.report.training_time + update.report.comm_time
                for update in self.updates
            )
            self.comm_time = max(update.report.comm_time for update in self.updates)

    async def process_client_info(self, client_id, sid):
        """Process the received metadata information from a reporting client."""
        if self.do_personalization_test:
            client_info = {
                "client_id": client_id,
                "sid": sid,
                "report": self.reports[sid],
                "payload": self.client_payload[sid],
            }
            await self.process_clients(client_info)
        else:
            await super().process_client_info(client_id, sid)

    async def _process_clients(self, client_info):
        """Process client reports."""

        if self.do_personalization_test:
            client = client_info
        else:
            client = client_info[2]
            client_staleness = self.current_round - client["starting_round"]

            self.updates.append((client["report"], client["payload"], client_staleness))

        if client["payload"] == "personalization_accuracy":
            self.personalization_test_updates.append(client["report"])

        if (
            Config().is_edge_server()
            and self.current_round <= Config().algorithm.local_rounds
            and self.current_round != 0
        ):
            # An edge server does not conduct personalization test until sending
            # its aggregated update to the central server
            # self.current_round == 0 means it just sent its aggregated update
            # to the central server
            if len(self.updates) > 0 and len(self.updates) >= len(
                self.selected_clients
            ):
                logging.info(
                    "[%s] All %d client reports received. Processing.",
                    self,
                    len(self.updates),
                )
                await super()._process_reports()
                await self.wrap_up()
                await self._select_clients()

        else:
            if len(self.personalization_test_updates) == 0:
                if len(self.updates) > 0 and len(self.updates) >= len(
                    self.selected_clients
                ):
                    logging.info(
                        "[%s] All %d client reports received. Processing.",
                        self,
                        len(self.updates),
                    )
                    await self._process_reports()

                    if Config().is_central_server():
                        # Start testing the global meta model w.r.t. personalization
                        await self.select_testing_clients()

            else:
                if len(self.personalization_test_updates) >= len(self.selected_clients):
                    logging.info(
                        "[%s] All %d personalization test results received.",
                        self,
                        len(self.personalization_test_updates),
                    )

                    await self._process_reports()

                    await self.wrap_up()
                    self.personalization_test_updates = []
                    self.do_personalization_test = False

                    # Start a new round of FL training
                    if Config().is_central_server():
                        await self._select_clients()
