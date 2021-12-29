"""
A cross-silo personalized federated learning server using MAML algorithm,
as either edge or central servers.
"""

import asyncio
import logging
import os
import pickle
import sys
import time

from plato.config import Config
from plato.utils import csv_processor
from plato.servers import fedavg_cs


class Server(fedavg_cs.Server):
    """Cross-silo federated learning server using federated averaging."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

        self.do_personalization_test = False

        # A list to store accuracy of clients' personalized models
        self.personalization_test_updates = []
        self.personalization_accuracy = 0

        self.training_time = 0

        if Config().is_edge_server():
            # An edge client waits for the event that a certain number of clients
            # compute accuracy of their personalized models
            self.per_accuracy_aggregated = asyncio.Event()

    async def select_testing_clients(self):
        """Select a subset of the clients to test personalization."""
        self.do_personalization_test = True
        logging.info("\n[Server #%d] Starting testing personalization.",
                     os.getpid())

        self.current_round -= 1
        await super().select_clients()

        if len(self.selected_clients) > 0:
            logging.info(
                "[Server #%d] Sent the current meta model to %d clients for personalization test.",
                os.getpid(), len(self.selected_clients))

    async def customize_server_response(self, server_response):
        """Wrap up generating the server response with any additional information."""
        if self.do_personalization_test:
            server_response['personalization_test'] = True
        return server_response

    async def process_reports(self):
        """Process the client reports by aggregating their weights."""
        if self.do_personalization_test:
            self.compute_personalization_accuracy()
            await self.wrap_up_processing_reports()
        else:
            await super().process_reports()

    def compute_personalization_accuracy(self):
        """"Average accuracy of clients' personalized models."""
        accuracy = 0
        for report in self.personalization_test_updates:
            accuracy += report
        self.personalization_accuracy = accuracy / len(
            self.personalization_test_updates)

    async def wrap_up_processing_reports(self):
        """Wrap up processing the reports with any additional work."""
        if self.do_personalization_test or Config().is_edge_server():
            if hasattr(Config(), 'results'):
                new_row = []
                for item in self.recorded_items:
                    item_value = {
                        'global_round':
                        self.current_global_round,
                        'round':
                        self.current_round,
                        'accuracy':
                        self.accuracy * 100,
                        'personalization_accuracy':
                        self.personalization_accuracy * 100,
                        'edge_agg_num':
                        Config().algorithm.local_rounds,
                        'local_epoch_num':
                        Config().trainer.epochs,
                        'training_time':
                        self.training_time,
                        'round_time':
                        time.perf_counter() - self.round_start_time
                    }[item]
                    new_row.append(item_value)

                if Config().is_edge_server():
                    result_csv_file = f'{Config().result_dir}result_{Config().args.id}.csv'
                else:
                    result_csv_file = f'{Config().result_dir}result.csv'

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
                            '[Server #%d] Completed %s rounds of local aggregation.',
                            os.getpid(),
                            Config().algorithm.local_rounds)
                        self.model_aggregated.set()

                        self.current_round = 0

        if not self.do_personalization_test:
            self.training_time = max(
                [report.training_time for (report, __) in self.updates])

    async def client_payload_done(self, sid, client_id, s3_key=None):
        """ Upon receiving all the payload from a client, eithe via S3 or socket.io. """
        if s3_key is None:
            assert self.client_payload[sid] is not None

            payload_size = 0
            if isinstance(self.client_payload[sid], list):
                for _data in self.client_payload[sid]:
                    payload_size += sys.getsizeof(pickle.dumps(_data))
            else:
                payload_size = sys.getsizeof(
                    pickle.dumps(self.client_payload[sid]))
        else:
            self.client_payload[sid] = self.s3_client.receive_from_s3(s3_key)
            payload_size = sys.getsizeof(pickle.dumps(
                self.client_payload[sid]))

        logging.info(
            "[Server #%d] Received %s MB of payload data from client #%d.",
            os.getpid(), round(payload_size / 1024**2, 2), client_id)

        if self.client_payload[sid] == 'personalization_accuracy':
            self.personalization_test_updates.append(self.reports[sid])
        else:
            self.updates.append((self.reports[sid], self.client_payload[sid]))

        if Config().is_edge_server() and self.current_round <= Config(
        ).algorithm.local_rounds and self.current_round != 0:
            # An edge server does not conduct personalization test until sending
            # its aggregated update to the central server
            # self.current_round == 0 means it just sent its aggregated update
            # to the central server
            if len(self.updates) > 0 and len(self.updates) >= len(
                    self.selected_clients):
                logging.info(
                    "[Edge Server #%d] All %d client reports received. Processing.",
                    os.getpid(), len(self.updates))
                await super().process_reports()
                await self.wrap_up()
                await self.select_clients()

        else:
            if len(self.personalization_test_updates) == 0:
                if len(self.updates) > 0 and len(self.updates) >= len(
                        self.selected_clients):
                    logging.info(
                        "[Server #%d] All %d client reports received. Processing.",
                        os.getpid(), len(self.updates))
                    await self.process_reports()

                    if Config().is_central_server():
                        # Start testing the global meta model w.r.t. personalization
                        await self.select_testing_clients()

            else:
                if len(self.personalization_test_updates) >= len(
                        self.selected_clients):
                    logging.info(
                        "[Server #%d] All %d personalization test results received.",
                        os.getpid(), len(self.personalization_test_updates))

                    await self.process_reports()

                    await self.wrap_up()
                    self.personalization_test_updates = []
                    self.do_personalization_test = False

                    # Start a new round of FL training
                    if Config().is_central_server():
                        await self.select_clients()
