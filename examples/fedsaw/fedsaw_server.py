"""
A cross-silo federated learning server using FedSaw,
as either central or edge servers.
"""

import logging
import math
import os
import pickle
import statistics
import sys

import torch

from plato.config import Config
from plato.servers import fedavg_cs


class Server(fedavg_cs.Server):
    """Cross-silo federated learning server using FedSaw."""

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

        # The central server uses a list to store each institution's clients' pruning amount
        self.pruning_amount_list = None
        self.comm_overhead = 0

        if Config().is_central_server():
            self.pruning_amount_list = [
                Config().clients.pruning_amount
                for i in range(Config().algorithm.total_silos)
            ]

        if Config().is_edge_server() and hasattr(Config(), 'results'):
            if 'pruning_amount' not in self.recorded_items:
                self.recorded_items = self.recorded_items + ['pruning_amount']

    async def customize_server_response(self, server_response):
        """ Wrap up generating the server response with any additional information. """
        server_response = await super().customize_server_response(
            server_response)

        if Config().is_central_server():
            server_response['pruning_amount'] = self.pruning_amount_list
        if Config().is_edge_server():
            # At this point, an edge server already updated Config().clients.pruning_amount
            # to the number received from the central server.
            # Now it could pass the new pruning amount to its clients.
            server_response['pruning_amount'] = Config().clients.pruning_amount

        return server_response

    def compute_weight_deltas(self, updates):
        """ Extract the model weight updates from client updates. """
        deltas_received = [payload for (__, __, payload, __) in updates]
        return deltas_received

    def update_pruning_amount_list(self):
        """ Update the list of each institution's clients' pruning_amount. """
        weights_diff_list = self.get_weights_differences()

        self.compute_pruning_amount(weights_diff_list)

    def compute_pruning_amount(self, weights_diff_list):
        """ A method to compute pruning amount. """

        median = statistics.median(weights_diff_list)

        for i, weight_diff in enumerate(weights_diff_list):
            if weight_diff >= median:
                self.pruning_amount_list[i] = Config(
                ).clients.pruning_amount * (
                    1 + math.tanh(weight_diff / sum(weights_diff_list)))
            else:
                self.pruning_amount_list[i] = Config(
                ).clients.pruning_amount * (
                    1 - math.tanh(weight_diff / sum(weights_diff_list)))

    def get_weights_differences(self):
        """
        Get the weights differences of each edge server's aggregated model
        and the global model.
        """
        weights_diff_list = []
        for i in range(Config().algorithm.total_silos):
            if hasattr(Config().clients,
                       'simulation') and Config().clients.simulation:
                client_id = i + 1 + Config().clients.per_round
            else:
                client_id = i + 1 + Config().clients.total_clients
            (report, received_updates) = [
                (report, payload) for (__, report, payload, __) in self.updates
                if int(report.client_id) == client_id
            ][0]
            num_samples = report.num_samples

            weights_diff = self.compute_weights_difference(
                received_updates, num_samples)

            weights_diff_list.append(weights_diff)

        return weights_diff_list

    def compute_weights_difference(self, received_updates, num_samples):
        """
        Compute the weight difference of an edge server's aggregated model
        and the global model.
        """
        weights_diff = 0

        for _, delta in received_updates.items():
            delta = delta.float()
            weights_diff += torch.norm(delta).item()

        weights_diff = weights_diff * (num_samples / self.total_samples)

        return weights_diff

    def get_record_items_values(self):
        """Get values will be recorded in result csv file."""
        record_items_values = super().get_record_items_values()
        record_items_values['pruning_amount'] = Config().clients.pruning_amount

        if Config().is_central_server():
            edge_comm_overhead = sum([
                report.comm_overhead for (__, report, __, __) in self.updates
            ])
            record_items_values[
                'comm_overhead'] = edge_comm_overhead + self.comm_overhead
        else:
            record_items_values['comm_overhead'] = self.comm_overhead

        return record_items_values

    async def wrap_up_processing_reports(self):
        """Wrap up processing the reports with any additional work."""
        await super().wrap_up_processing_reports()

        if Config().is_central_server():
            self.update_pruning_amount_list()
            self.comm_overhead = 0

    async def send(self, sid, payload, client_id) -> None:
        """ Sending a new data payload to the client using either S3 or socket.io. """
        # First apply outbound processors, if any
        payload = self.outbound_processor.process(payload)

        metadata = {'id': client_id}

        if self.s3_client is not None:
            s3_key = f'server_payload_{os.getpid()}_{self.current_round}'
            self.s3_client.send_to_s3(s3_key, payload)
            data_size = sys.getsizeof(pickle.dumps(payload))
            metadata['s3_key'] = s3_key
        else:
            data_size = 0

            if isinstance(payload, list):
                for data in payload:
                    _data = pickle.dumps(data)
                    await self.send_in_chunks(_data, sid, client_id)
                    data_size += sys.getsizeof(_data)

            else:
                _data = pickle.dumps(payload)
                await self.send_in_chunks(_data, sid, client_id)
                data_size = sys.getsizeof(_data)

        await self.sio.emit('payload_done', metadata, room=sid)

        logging.info("[%s] Sent %.2f MB of payload data to client #%d.", self,
                     data_size / 1024**2, client_id)
        self.comm_overhead += data_size / 1024**2

    async def client_report_arrived(self, sid, client_id, report):
        """ Upon receiving a report from a client. """
        self.reports[sid] = pickle.loads(report)
        self.client_payload[sid] = None
        self.client_chunks[sid] = []

        if self.comm_simulation:
            model_name = Config().trainer.model_name if hasattr(
                Config().trainer, 'model_name') else 'custom'
            checkpoint_path = Config().params['checkpoint_path']
            payload_filename = f"{checkpoint_path}/{model_name}_client_{client_id}.pth"
            with open(payload_filename, 'rb') as payload_file:
                self.client_payload[sid] = pickle.load(payload_file)

            data_size = sys.getsizeof(pickle.dumps(self.client_payload[sid]))
            logging.info(
                "[%s] Received %.2f MB of payload data from client #%d (simulated).",
                self, data_size / 1024**2, client_id)

            self.comm_overhead += data_size / 1024**2

            await self.process_client_info(client_id, sid)

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

        logging.info("[%s] Received %.2f MB of payload data from client #%d.",
                     self, payload_size / 1024**2, client_id)

        self.comm_overhead += payload_size / 1024**2

        await self.process_client_info(client_id, sid)
