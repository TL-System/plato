"""
A cross-silo federated learning server using Sub-FedAvg(Un),
as either central or edge servers.
"""

import logging
import os
import pickle
import sys

from plato.config import Config
from plato.servers import fedavg_cs


class Server(fedavg_cs.Server):
    """Cross-silo federated learning server using Sub-FedAvg(Un)."""

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

        self.comm_overhead = 0

        if Config().is_edge_server() and hasattr(Config(), "results"):
            if "pruning_amount" not in self.recorded_items:
                self.recorded_items = self.recorded_items + ["pruning_amount"]

    def get_record_items_values(self):
        """Get values will be recorded in result csv file."""
        record_items_values = super().get_record_items_values()
        record_items_values["pruning_amount"] = Config().clients.pruning_amount

        if Config().is_central_server():
            edge_comm_overhead = sum(
                update.report.comm_overhead for update in self.updates
            )
            record_items_values["comm_overhead"] = (
                edge_comm_overhead + self.comm_overhead
            )
        else:
            record_items_values["comm_overhead"] = self.comm_overhead

        return record_items_values

    async def wrap_up_processing_reports(self):
        """Wrap up processing the reports with any additional work."""
        await super().wrap_up_processing_reports()

        if Config().is_central_server():
            self.comm_overhead = 0

    async def send(self, sid, payload, client_id) -> None:
        """Sending a new data payload to the client using either S3 or socket.io."""
        # First apply outbound processors, if any
        payload = self.outbound_processor.process(payload)

        metadata = {"id": client_id}

        if self.s3_client is not None:
            s3_key = f"server_payload_{os.getpid()}_{self.current_round}"
            self.s3_client.send_to_s3(s3_key, payload)
            data_size = sys.getsizeof(pickle.dumps(payload))
            metadata["s3_key"] = s3_key
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

        await self.sio.emit("payload_done", metadata, room=sid)

        logging.info(
            "[%s] Sent %.2f MB of payload data to client #%d.",
            self,
            data_size / 1024**2,
            client_id,
        )
        self.comm_overhead += data_size / 1024**2

    async def client_report_arrived(self, sid, client_id, report):
        """Upon receiving a report from a client."""
        self.reports[sid] = pickle.loads(report)
        self.client_payload[sid] = None
        self.client_chunks[sid] = []

        if self.comm_simulation:
            model_name = (
                Config().trainer.model_name
                if hasattr(Config().trainer, "model_name")
                else "custom"
            )
            checkpoint_path = Config().params["checkpoint_path"]
            payload_filename = f"{checkpoint_path}/{model_name}_client_{client_id}.pth"
            with open(payload_filename, "rb") as payload_file:
                self.client_payload[sid] = pickle.load(payload_file)

            data_size = sys.getsizeof(pickle.dumps(self.client_payload[sid]))
            logging.info(
                "[%s] Received %.2f MB of payload data from client #%d (simulated).",
                self,
                data_size / 1024**2,
                client_id,
            )

            self.comm_overhead += data_size / 1024**2

            await self.process_client_info(client_id, sid)

    async def client_payload_done(self, sid, client_id, s3_key=None):
        """Upon receiving all the payload from a client, either via S3 or socket.io."""
        if s3_key is None:
            assert self.client_payload[sid] is not None

            payload_size = 0
            if isinstance(self.client_payload[sid], list):
                for _data in self.client_payload[sid]:
                    payload_size += sys.getsizeof(pickle.dumps(_data))
            else:
                payload_size = sys.getsizeof(pickle.dumps(self.client_payload[sid]))
        else:
            self.client_payload[sid] = self.s3_client.receive_from_s3(s3_key)
            payload_size = sys.getsizeof(pickle.dumps(self.client_payload[sid]))

        logging.info(
            "[%s] Received %.2f MB of payload data from client #%d.",
            self,
            payload_size / 1024**2,
            client_id,
        )

        self.comm_overhead += payload_size / 1024**2

        await self.process_client_info(client_id, sid)
