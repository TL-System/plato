"""
A federated learning client at the edge server in a cross-silo training workload.
"""

from dataclasses import dataclass

import logging
import os
import pickle
import sys

from plato.clients import edge


@dataclass
class Report(edge.Report):
    """Report from an Axiothea edge server, to be sent to the central server."""


class Client(edge.Client):
    """A federated learning client at the edge server in a cross-silo training workload."""
    def __init__(self, server, algorithm=None, trainer=None):
        super().__init__(server, algorithm=algorithm, trainer=trainer)
        self.do_personalization_test = False

    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        if 'personalization_test' in server_response:
            self.do_personalization_test = True
        else:
            super().process_server_response(server_response)

    async def test_personalization(self):
        """Test personalization by passing the global meta model to its clients,
        and let them train their personlized models and test accuracy."""
        logging.info(
            "[Edge Server #%d] Passing the global meta model to its clients.",
            self.client_id)

        # Edge server select clients to conduct personalization test
        await self.server.select_testing_clients()

        # Wait for clients conducting personalization test
        await self.server.per_accuracy_aggregated.wait()
        self.server.per_accuracy_aggregated.clear()

        report = self.server.personalization_accuracy
        payload = 'personalization_accuracy'

        return report, payload

    async def payload_done(self, client_id, object_key) -> None:
        """Upon receiving all the new payload from the server."""
        payload_size = 0

        if object_key is None:
            if isinstance(self.server_payload, list):
                for _data in self.server_payload:
                    payload_size += sys.getsizeof(pickle.dumps(_data))
            elif isinstance(self.server_payload, dict):
                for key, value in self.server_payload.items():
                    payload_size += sys.getsizeof(pickle.dumps({key: value}))
            else:
                payload_size = sys.getsizeof(pickle.dumps(self.server_payload))
        else:
            self.server_payload = self.s3_client.receive_from_s3(object_key)
            payload_size = sys.getsizeof(pickle.dumps(self.server_payload))

        assert client_id == self.client_id

        logging.info(
            "[Client #%d] Received %s MB of payload data from the server.",
            client_id, round(payload_size / 1024**2, 2))

        self.load_payload(self.server_payload)
        self.server_payload = None

        if self.do_personalization_test:
            report, payload = await self.test_personalization()
            self.do_personalization_test = False
        else:
            report, payload = await self.train()
            logging.info(
                "[Server #%d] Model aggregated on edge server (client #%d).",
                os.getpid(), client_id)

        # Sending the client report as metadata to the server (payload to follow)
        await self.sio.emit('client_report', {'report': pickle.dumps(report)})

        # Sending the client training payload to the server
        await self.send(payload)
