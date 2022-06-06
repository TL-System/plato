import logging
import pickle
import sys
import uuid
from dataclasses import dataclass

from plato.clients import base, simple
from plato.config import Config


@dataclass
class Report(base.Report):
    """Report from a simple client, to be sent to the federated learning server."""
    comm_time: float
    update_response: bool


class Client(simple.Client):
    def __init__(self, model, trainer):
        super().__init__(model=model, trainer=trainer)

    async def send(self, payload) -> None:
        """Sending the client payload to the server using simulation, S3 or socket.io."""
        file_path = f"{Config().params['model_path']}/{self.client_id}.pickle"
        with open(file_path, 'rb') as handle:
            payload = (payload, pickle.load(handle))

        # First apply outbound processors, if any
        payload = self.outbound_processor.process(payload)

        if self.comm_simulation:
            # If we are using the filesystem to simulate communication over a network
            model_name = Config().trainer.model_name if hasattr(
                Config().trainer, 'model_name') else 'custom'
            checkpoint_path = Config().params['checkpoint_path']
            payload_filename = f"{checkpoint_path}/{model_name}_client_{self.client_id}.pth"
            with open(payload_filename, 'wb') as payload_file:
                pickle.dump(payload, payload_file)

            logging.info(
                "[%s] Sent %.2f MB of payload data to the server (simulated).",
                self,
                sys.getsizeof(pickle.dumps(payload)) / 1024**2)
        else:
            metadata = {'id': self.client_id}

            if self.s3_client is not None:
                unique_key = uuid.uuid4().hex[:6].upper()
                s3_key = f'client_payload_{self.client_id}_{unique_key}'
                self.s3_client.send_to_s3(s3_key, payload)
                data_size = sys.getsizeof(pickle.dumps(payload))
                metadata['s3_key'] = s3_key
            else:
                if isinstance(payload, list):
                    data_size: int = 0

                    for data in payload:
                        _data = pickle.dumps(data)
                        await self.send_in_chunks(_data)
                        data_size += sys.getsizeof(_data)
                else:
                    _data = pickle.dumps(payload)
                    await self.send_in_chunks(_data)
                    data_size = sys.getsizeof(_data)

            await self.sio.emit('client_payload_done', metadata)

            logging.info("[%s] Sent %.2f MB of payload data to the server.",
                         self, data_size / 1024**2)
