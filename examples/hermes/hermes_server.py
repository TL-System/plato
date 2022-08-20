"""
A federated learning server using Hermes.
"""

import copy
import logging
import os
import pickle
import sys
import numpy as np
import torch

import hermes_pruning as pruning
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

        # Get the list of client models and masks
        weights_received, masks_received = self.extract_client_updates(updates)

        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (__, report, __, __) in updates]
        )

        # Perform averaging of overlapping parameters
        for layer_name in weights_received[0].keys():
            for model in weights_received:
                model[layer_name] = model[layer_name].numpy()

        step = 0

        masked_layers = []
        for name, module in self.trainer.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                layer_name = f"{name}.weight"
                masked_layers.append(layer_name)

        for layer_name in weights_received[0].keys():
            if layer_name in masked_layers:
                count = np.zeros_like(masks_received[0][step].reshape([-1]))
                avg = np.zeros_like(weights_received[0][layer_name].reshape([-1]))
                for index, __ in enumerate(masks_received):
                    __, report, __, __ = updates[index]
                    num_samples = report.num_samples
                    count += masks_received[index][step].reshape([-1])
                    avg += weights_received[index][layer_name].reshape([-1]) * (
                        num_samples / self.total_samples
                    )

                count = np.where(count == len(masks_received), 1, 0)
                final_avg = np.divide(avg, count)
                ind = np.isfinite(final_avg)

                for model in weights_received:
                    model[layer_name].reshape([-1])[ind] = final_avg[ind]
                    shape = weights_received[0][layer_name].shape
                    model[layer_name] = torch.from_numpy(
                        model[layer_name].reshape(shape)
                    )
                step = step + 1
            else:
                avg = np.zeros_like(weights_received[0][layer_name].reshape([-1]))
                if "int" in str(avg.dtype):
                    avg = avg.astype(np.float64)
                for index, __ in enumerate(weights_received):
                    __, report, __, __ = updates[index]
                    num_samples = report.num_samples
                    avg += weights_received[index][layer_name].reshape([-1]) * (
                        num_samples / self.total_samples
                    )

                shape = weights_received[0][layer_name].shape
                new_tensor = torch.from_numpy(avg.reshape(shape))
                for model in weights_received:
                    model[layer_name] = new_tensor

        return weights_received

    async def process_reports(self):
        """Process the client reports by aggregating their overlapping weights."""
        self.personalized_models = await self.federated_averaging(self.updates)
        self.accuracy = self.accuracy_averaging(self.updates)
        logging.info("[%s] Average client accuracy: %.2f%%.", self, 100 * self.accuracy)
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

    def process_customized_report(self, client_id, checkpoint_path, model_name):
        """Process a customized client report with additional information."""
        # Include the pruning mask in the communication overhead
        mask_filename = f"{checkpoint_path}/{model_name}_client{client_id}_mask.pth"
        if os.path.exists(mask_filename):
            with open(mask_filename, "rb") as payload_file:
                client_mask = pickle.load(payload_file)
                mask_size = sys.getsizeof(pickle.dumps(client_mask)) / 1024**2
                logging.info(
                    "[%s] Received %.2f MB of pruning mask from client #%d (simulated).",
                    self,
                    mask_size,
                    client_id,
                )

                self.comm_overhead += mask_size

                self.uplink_comm_time[client_id] += mask_size / (
                    self.uplink_bandwidth / 8
                )

    def customize_server_payload(self, payload):
        """Wrap up generating the server payload with any additional information."""

        # If the client has already begun the learning of a personalized model
        # in a previous communication round, the personalized file is loaded and
        # sent to the client for continued training. Otherwise, if the client is
        # selected for the first time, it receives the pre-initialized model.
        model_name = (
            Config().trainer.model_name
            if hasattr(Config().trainer, "model_name")
            else "custom"
        )
        model_path = Config().params["model_path"]
        if not self.clients_first_time[self.selected_client_id - 1]:
            filename = (
                f"{model_path}/personalized_{model_name}"
                f"_client{self.selected_client_id}.pth"
            )
            with open(filename, "rb") as payload_file:
                payload = pickle.load(payload_file)
        else:
            self.clients_first_time[self.selected_client_id - 1] = False

        return payload

    def extract_client_updates(self, updates):
        """Extract the model weight updates from client updates along with the masks."""
        model_name = Config().trainer.model_name
        checkpoint_path = Config().params["checkpoint_path"]

        weights_received = [payload for (__, __, payload, __) in updates]

        masks_received = []
        for (client_id, __, payload, __) in updates:
            mask_path = f"{checkpoint_path}/{model_name}_client{client_id}_mask.pth"
            if os.path.exists(mask_path):
                with open(mask_path, "rb") as mask_file:
                    masks_received.append(pickle.load(mask_file))
            else:
                model = copy.deepcopy(self.algorithm.model)
                model.load_state_dict(payload, strict=True)
                mask = pruning.make_init_mask(model)
                masks_received.append(mask)

        return weights_received, masks_received

    def server_will_close(self):
        """Method called at the start of closing the server."""
        # Delete pruning masks created by clients
        model_name = Config().trainer.model_name
        checkpoint_path = Config().params["checkpoint_path"]
        for client_id in range(1, self.total_clients + 1):
            mask_path = f"{checkpoint_path}/{model_name}_client{client_id}_mask.pth"
            if os.path.exists(mask_path):
                os.remove(mask_path)
