"""
A personalized federated learning client using MAML algorithm for local training.
"""
import logging
import pickle
import sys
from dataclasses import dataclass

from plato.clients import simple


@dataclass
class Report(simple.Report):
    """A client report."""


class Client(simple.Client):
    """A federated learning client."""
    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model=model,
                         datasource=datasource,
                         algorithm=algorithm,
                         trainer=trainer)
        self.do_personalization_test = False

    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        if 'personalization_test' in server_response:
            self.do_personalization_test = True

    async def payload_done(self, client_id, object_key) -> None:
        """ Upon receiving all the new payload from the server. """
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
            # Train a personalized model based on the current meta model and test it
            # This report only contains accuracy of its personalized model
            report = await self.test_personalized_model()
            payload = 'personalization_accuracy'
            self.do_personalization_test = False
        else:
            # Regular local training of FL
            report, payload = await self.train()
            logging.info("[Client #%d] Model trained.", client_id)

        # Sending the client report as metadata to the server (payload to follow)
        await self.sio.emit('client_report', {'report': pickle.dumps(report)})

        # Sending the client training payload to the server
        await self.send(payload)

    async def test_personalized_model(self):
        """A client first trains its personalized model based on
        the global meta model and then test it.
        """
        logging.info("[Client #%d] Started training a personalized model.",
                     self.client_id)

        # Train a personalized model and test it
        self.trainer.test_personalization = True
        personalization_accuracy = self.trainer.test(self.testset)
        self.trainer.test_personalization = False

        if personalization_accuracy == 0:
            # The testing process failed, disconnect from the server
            await self.sio.disconnect()

        logging.info("[Client #{:d}] Personlization accuracy: {:.2f}%".format(
            self.client_id, 100 * personalization_accuracy))

        return personalization_accuracy
