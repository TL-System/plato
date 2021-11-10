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
        self.do_meta_personalization_test = False
        self.do_local_personalization = False

        # A list to store accuracy of clients' personalized models with meta-learning
        self.meta_personalization_test_updates = []
        self.meta_personalization_accuracy = 0

        self.training_time = 0

    async def select_testing_clients(self):
        """Select a subset of the clients to test personalization."""

        self.do_meta_personalization_test = True

        logging.info("\n[Server #%d] Starting meta testing personalization.",
                     os.getpid())

        self.current_round -= 1
        # set the clients used in next round for testing
        self.clients_per_round = Config().clients.test_clients
        await super().select_clients()
        # after selection, change it back
        self.clients_per_round = Config().clients.per_round

        if len(self.selected_clients) > 0:
            logging.info(
                "[Server #%d] Sent the current meta model to %d clients for personalization test.",
                os.getpid(), len(self.selected_clients))

    async def customize_server_response(self, server_response):
        """Wrap up generating the server response with any additional information."""
        if self.do_meta_personalization_test:
            server_response['meta_personalization_test'] = True

        return server_response

    async def process_reports(self):
        """Process the client reports by aggregating their weights."""
        if self.do_meta_personalization_test:
            self.meta_personalization_accuracy = self.compute_personalization_accuracy(
                self.meta_personalization_test_updates)
            await self.wrap_up_processing_reports()
        else:
            await self.aggregate_weights(self.updates)
            await self.wrap_up_processing_reports()

    def compute_personalization_accuracy(self, personalization_test_updates):
        """"Average accuracy of clients' personalized models."""
        accuracy = 0
        total_reports = personalization_test_updates
        total_num_reported_clients = len(total_reports)
        for report in total_reports:
            accuracy += report
        averaged_acc = accuracy / total_num_reported_clients
        return averaged_acc

    async def wrap_up_processing_reports(self):
        """Wrap up processing the reports with any additional work."""
        if self.do_meta_personalization_test:
            if hasattr(Config(), 'results'):
                new_row = []
                for item in self.recorded_items:
                    item_value = {
                        'round':
                        self.current_round,
                        'accuracy':
                        self.meta_personalization_accuracy * 100,
                        'personalization_accuracy':
                        self.meta_personalization_accuracy * 100,
                        'training_time':
                        self.training_time,
                        'round_time':
                        time.perf_counter() - self.round_start_time
                    }[item]
                    new_row.append(item_value)
                test_name = "meta_personalization"
                save_file_name = f"{self.current_round}_{test_name}_{Config().params['run_id']}.pth"
                result_csv_file = os.path.join(Config().result_dir,
                                               save_file_name + '_result.csv')

                csv_processor.write_csv(result_csv_file, new_row)

            self.do_meta_personalization_test = False
        else:
            self.training_time = max(
                [report.training_time for (report, __) in self.updates])

    async def client_payload_done(self, sid, client_id, object_key):
        """ Upon receiving all the payload from a client, either via S3 or socket.io. """
        if object_key is None:
            assert self.client_payload[sid] is not None

            payload_size = 0
            if isinstance(self.client_payload[sid], list):
                for _data in self.client_payload[sid]:
                    payload_size += sys.getsizeof(pickle.dumps(_data))
            else:
                payload_size = sys.getsizeof(
                    pickle.dumps(self.client_payload[sid]))
        else:
            self.client_payload[sid] = self.s3_client.receive_from_s3(
                object_key)
            payload_size = sys.getsizeof(pickle.dumps(
                self.client_payload[sid]))

        logging.info(
            "[Server #%d] Received %s MB of payload data from client #%d.",
            os.getpid(), round(payload_size / 1024**2, 2), client_id)

        if self.client_payload[sid] == 'meta_personalization_accuracy':
            self.meta_personalization_test_updates.append(self.reports[sid])
        else:
            self.updates.append((self.reports[sid], self.client_payload[sid]))

        logging.info(
            "[Server #%d] Received %s MB of payload data from client #%d.",
            os.getpid(), round(payload_size / 1024**2, 2), client_id)

        if len(self.meta_personalization_test_updates) == 0:
            if len(self.updates) > 0 and len(self.updates) >= len(
                    self.selected_clients):
                logging.info(
                    "[Server #%d] All %d client reports received. Processing.",
                    os.getpid(), len(self.updates))
                self.do_meta_personalization_test = False
                await self.process_reports()

                # Start testing the global meta model w.r.t. personalization
                await self.select_testing_clients()

        if len(self.meta_personalization_test_updates) > 0 and len(
                self.meta_personalization_test_updates) >= len(
                    self.selected_clients):
            logging.info(
                "[Server #%d] All %d personalization test results received.",
                os.getpid(), len(self.meta_personalization_test_updates))

            await self.process_reports()

            await self.wrap_up()
            self.meta_personalization_test_updates = []

            # Start a new round of FL training
            await self.select_clients()
