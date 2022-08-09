"""
A federated learning client of Hermes.
"""

import logging
import os
import pickle
import sys

from plato.config import Config
from plato.clients import simple


class Client(simple.Client):
    """
    A federated learning client of Hermes.
    """

    async def send(self, payload) -> None:
        """Sending the client payload to the server using simulation, S3 or socket.io."""
        super().send(payload)

        if self.comm_simulation:
            # If we are using the filesystem to simulate communication over a network
            model_name = (
                Config().trainer.model_name
                if hasattr(Config().trainer, "model_name")
                else "custom"
            )
            checkpoint_path = Config().params["checkpoint_path"]

            mask_filename = (
                f"{checkpoint_path}/{model_name}_client{self.client_id}_mask.pth"
            )
            if os.path.exists(mask_filename):
                with open(mask_filename, "rb") as payload_file:
                    client_mask = pickle.load(payload_file)
                    mask_size = sys.getsizeof(pickle.dumps(client_mask)) / 1024**2
            else:
                mask_size = 0

            logging.info(
                "[%s] Sent %.2f MB of pruning mask to the server (simulated).",
                self,
                mask_size,
            )
