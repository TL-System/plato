"""
A federated learning client at the edge server in a cross-silo training workload.
"""

from dataclasses import dataclass
import logging
import pickle

from plato.clients import edge


@dataclass
class Report(edge.Report):
    """Report from an Axiothea edge server, to be sent to the central server."""


class Client(edge.Client):
    """A federated learning client at the edge server in a cross-silo training workload."""

    def __init__(self, server):
        super().__init__(server)
        self.do_personalization_test = False

    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        if "personalization_test" in server_response:
            self.do_personalization_test = True
        else:
            super().process_server_response(server_response)

    async def test_personalization(self):
        """Test personalization by passing the global meta model to its clients,
        and let them train their personlized models and test accuracy."""
        logging.info(
            "[Edge Server #%d] Passing the global meta model to its clients.",
            self.client_id,
        )

        # Edge server select clients to conduct personalization test
        await self.server.select_testing_clients()

        # Wait for clients conducting personalization test
        await self.server.per_accuracy_aggregated.wait()
        self.server.per_accuracy_aggregated.clear()

        report = self.server.personalization_accuracy
        payload = "personalization_accuracy"

        return report, payload

    async def _start_training(self):
        """Complete one round of training on this client."""
        self._load_payload(self.server_payload)
        self.server_payload = None

        if self.do_personalization_test:
            report, payload = await self.test_personalization()
            self.do_personalization_test = False
        else:
            report, payload = await self.train()
            logging.info("[%s] Model aggregated on edge server.", self)

        # Sending the client report as metadata to the server (payload to follow)
        await self.sio.emit(
            "client_report", {"id": self.client_id, "report": pickle.dumps(report)}
        )

        # Sending the client training payload to the server
        await self.send(payload)
