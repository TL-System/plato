"""
A federated learning server using Hermes.
"""

import numpy as np
import torch

import hermes_pruning as pruning

from plato.servers import fedavg_personalized as personalized_server


class Server(personalized_server.Server):
    """A federated learning server using the Hermes algorithm."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )
        self.masks_received = []
        self.aggregated_clients_model = {}
        self.total_samples = 0

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Aggregates weight updates from the clients using personalized aggregating."""
        # Extract the total number of samples
        self.total_samples = sum(update.report.num_samples for update in updates)

        # Perform averaging of overlapping parameters
        for layer_name in weights_received[0].keys():
            for model in weights_received:
                model[layer_name] = model[layer_name].numpy()

        step = 0

        masked_layers = []
        for name, layer in self.trainer.model.named_parameters():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                layer_name = f"{name}.weight"
                masked_layers.append(layer_name)

        for layer_name in weights_received[0].keys():
            if layer_name in masked_layers:
                count = np.zeros_like(self.masks_received[0][step].reshape([-1]))
                avg = np.zeros_like(weights_received[0][layer_name].reshape([-1]))
                for index, __ in enumerate(self.masks_received):
                    num_samples = updates[index].report.num_samples
                    count += self.masks_received[index][step].reshape([-1])
                    avg += weights_received[index][layer_name].reshape([-1]) * (
                        num_samples / self.total_samples
                    )

                count = np.where(count == len(self.masks_received), 1, 0)
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
                for index, weight_received in enumerate(weights_received):
                    num_samples = updates[index].report.num_samples
                    avg += weight_received[layer_name].reshape([-1]) * (
                        num_samples / self.total_samples
                    )

                shape = weights_received[0][layer_name].shape
                new_tensor = torch.from_numpy(avg.reshape(shape))
                for model in weights_received:
                    model[layer_name] = new_tensor

        self.update_client_model(weights_received, updates)

        deltas_received = self.algorithm.compute_weight_deltas(
            baseline_weights, weights_received
        )

        deltas = await self.aggregate_deltas(self.updates, deltas_received)
        # Updates the existing model weights from the provided deltas
        updated_weights = self.algorithm.update_weights(deltas)

        return updated_weights

    def update_client_model(self, aggregated_clients_models, updates):
        """Update clients' models."""
        for client_model, update in zip(aggregated_clients_models, updates):
            received_client_id = update.client_id
            if received_client_id in self.aggregated_clients_model:
                self.aggregated_clients_model[received_client_id] = client_model

    def customize_server_payload(self, payload):
        """Customizes the server payload before sending to the client."""

        # If the client has already begun training a personalized model
        # in a previous communication round, the personalized file is loaded and
        # sent to the client for continued training. Otherwise, if the client is
        # selected for the first time, it receives the pre-initialized model.
        if self.selected_client_id in self.aggregated_clients_model:
            # replace the payload for the current client with the personalized model
            payload = self.aggregated_clients_model[self.selected_client_id]

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
