"""
A federated learning server using Hermes.
"""

import logging
import os
import pickle
import sys
import numpy as np
import torch

from plato.servers import fedavg
from plato.config import Config


class Server(fedavg.Server):
    """A federated learning server using the Hermes algorithm."""

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        self.clients_first_time = [True for _ in range(Config().clients.total_clients)]
        self.personalized_models = []

    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using personalized aggregating."""
        weights_received = [payload for (__, __, payload, __) in updates]

        # Perform averaging of overlapping parameters
        for layer_name in sorted(weights_received[0].keys()):
            for model in weights_received:
                model[layer_name] = model[layer_name].numpy()

        for layer_name in sorted(weights_received[0].keys()):
            if "weight" in layer_name:
                for index, __ in np.ndenumerate(weights_received[0][layer_name]):
                    values = []
                    for model in weights_received:
                        values.append(model[layer_name][index])
                    if 0 not in values:
                        average = sum(values) / len(values)
                        for model in weights_received:
                            model[layer_name][index] = average

        for layer_name in sorted(weights_received[0].keys()):
            for model in weights_received:
                model[layer_name] = torch.from_numpy(model[layer_name])

        return weights_received

    async def process_reports(self):
        """
        Process the client reports by aggregating their overlapping weights.
        Hermes does not compute a global model, so its accuracy is not calculated.
        """
        self.personalized_models = await self.federated_averaging(self.updates)
        self.save_personalized_models(self.personalized_models, self.updates)
        await self.wrap_up_processing_reports()

    def save_personalized_models(self, personalized_models, updates):
        """Save each client's personalized model."""
        for (personalized_model, (client_id, __, __, __)) in zip(
            personalized_models, updates
        ):
            model_name = (
                Config().trainer.model_name
                if hasattr(Config().trainer, "model_name")
                else "custom"
            )
            model_path = Config().params["model_path"]
            filename = f"{model_path}/personalized_{model_name}_client{client_id}.pth"
            with open(filename, "wb") as payload_file:
                pickle.dump(personalized_model, payload_file)
            logging.info(
                "[%s] Saved client #%d's personalized model in: %s",
                self,
                client_id,
                filename,
            )

    def received_client_report(self, client_id):
        """Method called at the end of receiving a report from a client."""
        model_name = (
            Config().trainer.model_name
            if hasattr(Config().trainer, "model_name")
            else "custom"
        )
        checkpoint_path = Config().params["checkpoint_path"]
        mask_filename = f"{checkpoint_path}/{model_name}_client{client_id}_mask.pth"
        if os.path.exists(mask_filename):
            with open(mask_filename, "rb") as payload_file:
                client_mask = pickle.load(payload_file)
                mask_size = sys.getsizeof(pickle.dumps(client_mask)) / 1024**2
        else:
            mask_size = 0

        if mask_size != 0:
            logging.info(
                "[%s] Received %.2f MB of pruning mask from client #%d (simulated).",
                self,
                mask_size,
                client_id,
            )

            self.comm_overhead += mask_size

            self.uplink_comm_time[client_id] += mask_size / (self.uplink_bandwidth / 8)

    def server_will_close(self):
        """Method called at the start of closing the server."""
        # Delete pruning masks created by clients
        model_name = Config().trainer.model_name
        model_path = Config().params["checkpoint_path"]
        for client_id in range(1, self.total_clients + 1):
            mask_path = f"{model_path}/{model_name}_client{client_id}_mask.pth"
            if os.path.exists(mask_path):
                os.remove(mask_path)
