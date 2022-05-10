"""
A federated learning server for personalized FL.
"""
import logging
import os
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

        self.round_time = 0
        self.comm_time = 0

    async def select_testing_clients(self):
        """Select a subset of the clients to test personalization."""
        self.do_personalization_test = True
        logging.info("\n[%s] Starting testing personalization.", self)

        self.current_round -= 1
        await super().select_clients()

        if len(self.selected_clients) > 0:
            logging.info(
                "[%s] Sent the current meta model to %d clients for personalization test.",
                self, len(self.selected_clients))

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
                        'elapsed_time':
                        self.wall_time - self.initial_wall_time,
                        'comm_time':
                        self.comm_time,
                        'round_time':
                        self.round_time,
                    }[item]
                    new_row.append(item_value)

                result_csv_file = f"{Config().params['result_dir']}/{os.getpid()}.csv"

                csv_processor.write_csv(result_csv_file, new_row)

            self.do_personalization_test = False

        else:
            self.round_time = max([
                report.training_time + report.comm_time
                for (__, report, __, __) in self.updates
            ])
            self.comm_time = max(
                [report.comm_time for (__, report, __, __) in self.updates])

    async def process_client_info(self, client_id, sid):
        """ Process the received metadata information from a reporting client. """
        if self.do_personalization_test:
            client_info = {
                'client_id': client_id,
                'sid': sid,
                'report': self.reports[sid],
                'payload': self.client_payload[sid],
            }
            await self.process_clients(client_info)
        else:
            await super().process_client_info(client_id, sid)

    async def process_clients(self, client_info):
        """ Process client reports. """

        if self.do_personalization_test:
            client = client_info
        else:
            client = client_info[1]
            client_staleness = self.current_round - client['starting_round']

            self.updates.append(
                (client['report'], client['payload'], client_staleness))

        if client['payload'] == 'personalization_accuracy':
            self.personalization_test_updates.append(client['report'])

        if len(self.personalization_test_updates) == 0:
            if len(self.updates) > 0 and len(self.updates) >= len(
                    self.selected_clients):
                logging.info(
                    "[%s] All %d client reports received. Processing.", self,
                    len(self.updates))
                self.do_personalization_test = False
                await self.process_reports()

                # Start testing the global meta model w.r.t. personalization
                await self.select_testing_clients()

        if len(self.personalization_test_updates) > 0 and len(
                self.personalization_test_updates) >= len(
                    self.selected_clients):
            logging.info("[%s] All %d personalization test results received.",
                         self, len(self.personalization_test_updates))

            await self.process_reports()

            await self.wrap_up()
            self.personalization_test_updates = []

            # Start a new round of FL training
            await self.select_clients()
