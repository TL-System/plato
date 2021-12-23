"""
A federated learning server for personalized FL.
"""
import logging
import os
import pickle
import sys
import time
from plato.config import Config
from plato.servers import fedavg
from plato.utils import csv_processor


class Server(fedavg.Server):
    """A federated learning server for personalized FL."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        self.do_personalization_test = False

        # A list to store accuracy of clients' personalized models
        self.personalization_test_updates = []
        self.personalization_accuracy = 0

        self.training_time = 0

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
        if self.do_personalization_test:
            if hasattr(Config(), 'results'):
                new_row = []
                for item in self.recorded_items:
                    item_value = {
                        'round':
                        self.current_round,
                        'accuracy':
                        self.accuracy * 100,
                        'personalization_accuracy':
                        self.personalization_accuracy * 100,
                        'training_time':
                        self.training_time,
                        'round_time':
                        time.perf_counter() - self.round_start_time
                    }[item]
                    new_row.append(item_value)

                result_csv_file = Config().result_dir + 'result.csv'

                csv_processor.write_csv(result_csv_file, new_row)

            self.do_personalization_test = False

        else:
            self.training_time = max(
                [report.training_time for (report, __) in self.updates])

    async def client_payload_done(self, sid, client_id, s3_key=None):
        """ Upon receiving all the payload from a client, either via S3 or socket.io. """
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

        if len(self.personalization_test_updates) == 0:
            if len(self.updates) > 0 and len(self.updates) >= len(
                    self.selected_clients):
                logging.info(
                    "[Server #%d] All %d client reports received. Processing.",
                    os.getpid(), len(self.updates))
                self.do_personalization_test = False
                await self.process_reports()

                # Start testing the global meta model w.r.t. personalization
                await self.select_testing_clients()

        if len(self.personalization_test_updates) > 0 and len(
                self.personalization_test_updates) >= len(
                    self.selected_clients):
            logging.info(
                "[Server #%d] All %d personalization test results received.",
                os.getpid(), len(self.personalization_test_updates))

            await self.process_reports()

            await self.wrap_up()
            self.personalization_test_updates = []

            # Start a new round of FL training
            await self.select_clients()
