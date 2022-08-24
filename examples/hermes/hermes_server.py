"""
A federated learning server using Hermes.
"""

import logging
import pickle
import numpy as np

import hermes_pruning as pruning
from plato.servers import fedavg
from plato.config import Config


class Server(fedavg.Server):
    """A federated learning server using the Hermes algorithm."""

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        self.clients_first_time = [True for _ in range(Config().clients.total_clients)]
        self.personalized_models = [None] * Config().clients.total_clients
        self.masks_received = []

    # pylint: disable=unused-argument
    def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Aggregate weight updates from the clients using personalized aggregating."""
        # Extract the total number of samples
        self.total_samples = sum(update.report.num_samples for update in updates)

        # Perform averaging of overlapping parameters
        for layer_name in weights_received[0].keys():
            for model in weights_received:
                model[layer_name] = model[layer_name].numpy()

        for layer_name in weights_received[0].keys():
            count = np.zeros_like(self.masks_received[0][layer_name].reshape([-1]))
            avg = np.zeros_like(weights_received[0][layer_name].reshape([-1]))
            for index, __ in enumerate(self.masks_received):
                num_samples = updates[index].report.num_samples
                count += self.masks_received[index][layer_name].reshape([-1])
                avg += weights_received[index][layer_name].reshape([-1]) * (
                    num_samples / self.total_samples
                )

            count = np.where(count == len(self.masks_received), 1, 0)
            final_avg = np.divide(avg, count)
            ind = np.isfinite(final_avg)

            for model in weights_received:
                model[layer_name].reshape([-1])[ind] = final_avg[ind]
                shape = weights_received[0][layer_name].shape
                model[layer_name] = self.trainer.from_numpy(
                    model[layer_name].reshape(shape)
                )

        step = 0
        for update in updates:
            self.personalized_models[update.client_id - 1] = weights_received[step]
            step += 1

        self.save_personalized_models(weights_received, updates)
        return baseline_weights

    def save_personalized_models(self, personalized_models, updates):
        """Save each client's personalized model at the end of aggregation."""
        for (personalized_model, update) in zip(personalized_models, updates):
            model_name = (
                Config().trainer.model_name
                if hasattr(Config().trainer, "model_name")
                else "custom"
            )
            model_path = Config().params["model_path"]
            filename = (
                f"{model_path}/personalized_{model_name}_client{update.client_id}.pth"
            )
            with open(filename, "wb") as payload_file:
                pickle.dump(personalized_model, payload_file)
            logging.info(
                "[%s] Saved client #%d's personalized model in: %s",
                self,
                update.client_id,
                filename,
            )

    def customize_server_payload(self, payload):
        """Customizes the server payload before sending to the client."""

        # If the client has already begun the learning of a personalized model
        # in a previous communication round, the personalized file is loaded and
        # sent to the client for continued training. Otherwise, if the client is
        # selected for the first time, it receives the pre-initialized model.

        if not self.clients_first_time[self.selected_client_id - 1]:
            payload = self.personalized_models[self.selected_client_id - 1]
            logging.info(
                "[%s] Loaded client #%d's personalized model",
                self,
                self.selected_client_id,
            )
        else:
            self.clients_first_time[self.selected_client_id - 1] = False

        return payload

    def weights_received(self, weights_received):
        """Event called after the updated weights have been received."""

        # Extract the model weight updates from client updates along with the masks
        self.masks_received = [payload[1] for payload in weights_received]
        weights = [payload[0] for payload in weights_received]
        for step, mask in enumerate(self.masks_received):
            if mask is None:
                mask = pruning.make_init_mask(self.trainer.model)
                self.masks_received[step] = mask

        return weights
