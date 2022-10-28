"""
A MaskCrypt client with selective homomorphic encryption support.
"""
import random
import time
import torch
import pickle

from plato.clients import simple
from plato.config import Config
import maskcrypt_utils


class Client(simple.Client):
    """A MaskCrypt client with selective homomorphic encryption support."""

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

        self.encrypt_ratio = Config().clients.encrypt_ratio
        self.random_mask = Config().clients.random_mask
        self.final_mask = None

        self.attack_prep_dir = f"{Config().data.datasource}_{Config().trainer.model_name}_{self.encrypt_ratio}"
        if self.random_mask:
            self.attack_prep_dir += "_random"

        self.checkpoint_path = Config().params["checkpoint_path"]
        self.model_buffer = {}

    async def inbound_processed(self, processed_inbound_payload):
        """
        Couduct training and compute mask in odd rounds, send updates in even rounds.
        """
        if self.current_round % 2 != 0:
            # Conduct local training and compute encryption mask after that
            report, model_weights = await super().inbound_processed(
                processed_inbound_payload
            )
            mask_proposal = self._compute_mask(
                self.algorithm.extract_weights(), self.trainer.get_gradient()
            )
            self.model_buffer[self.client_id] = (report, model_weights)
            return report, mask_proposal
        else:
            # Set final encryption mask and send model updates to server
            self.final_mask = processed_inbound_payload
            report, weights = self.model_buffer.pop(self.client_id)
            # Set training_time to a non-zero value to avoid heapq.heappush error in base.Server Line 886
            report.training_time = 0.001
            report.comm_time = time.time()
            return report, weights

    def _get_exposed_weights(self):
        """Get the exposed model weights so far."""
        model_name = Config().trainer.model_name
        est_filename = (
            self.checkpoint_path
            + f"/{self.attack_prep_dir}/{model_name}_est_{self.client_id}.pth"
        )
        return maskcrypt_utils.get_est(est_filename)

    def _compute_mask(self, latest_weights, gradients):
        """Compute the encryption mask for current client."""
        exposed_flat = self._get_exposed_weights()
        exposed_flat = torch.tensor(exposed_flat)
        device = exposed_flat.device

        latest_flat = torch.cat(
            [
                torch.flatten(latest_weights[name])
                for _, name in enumerate(latest_weights)
            ]
        )
        # Store the plain model weights
        plain_filename = (
            f"{self.checkpoint_path}/{self.attack_prep_dir}/"
            + f"{Config().trainer.model_name}_plain_{self.client_id}.pth"
        )
        with open(plain_filename, "wb") as plain_file:
            pickle.dump(latest_flat, plain_file)

        if self.random_mask:
            # Return a random mask when enabled in config file
            rand_mask = random.sample(
                range(len(exposed_flat)), int(self.encrypt_ratio * len(exposed_flat))
            )
            return torch.tensor(rand_mask)

        gradient_list = [
            torch.flatten(gradients[name]).to(device)
            for _, name in enumerate(gradients)
        ]
        grad_flat = torch.cat(gradient_list)
        delta = exposed_flat - latest_flat
        product = delta * grad_flat

        _, indices = torch.sort(product, descending=True)
        mask_len = int(self.encrypt_ratio * len(indices))

        return indices[:mask_len].clone().detach()
