"""
A simple federated learning server using federated averaging with homomorphic encryption support.
"""
import os
import pickle
import torch

from typing import OrderedDict
from plato.config import Config
from plato.servers import fedavg

import encrypt_utils as homo_enc


class Server(fedavg.Server):
    """Federated learning server using federated averaging with homomorphic encryption support."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(model, datasource, algorithm, trainer, callbacks)
        self.encrypted_model = None
        self.weight_shapes = {}
        self.para_nums = {}
        self.ckks_context = homo_enc.get_ckks_context()

        self.final_mask = None
        self.last_selected_clients = []

        self.param_inited = False

        self.checkpoint_path = Config().params["checkpoint_path"]
        self.attack_prep_dir = f"{Config().data.datasource}_{Config().trainer.model_name}_{Config().clients.encrypt_ratio}"
        if Config().clients.random_mask:
            self.attack_prep_dir += "_random"
        if not os.path.exists(f"{self.checkpoint_path}/{self.attack_prep_dir}/"):
            os.mkdir(f"{self.checkpoint_path}/{self.attack_prep_dir}/")

    def choose_clients(self, clients_pool, clients_count):
        """Choose the same clients every two rounds."""
        if self.current_round % 2 != 0:
            self.last_selected_clients = super().choose_clients(
                clients_pool, clients_count
            )
        return self.last_selected_clients

    def customize_server_payload(self, payload):
        """Customize the server payload before sending to the client."""
        if not self.param_inited:
            self._init_model_params()

        if self.current_round % 2 != 0:
            return self.encrypted_model
        else:
            return self.final_mask

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        if self.current_round % 2 != 0:
            self.mask_consensus(updates)
            return baseline_weights
        else:
            return self._aggregate(updates)

    def _init_model_params(self):
        """Initialize and save the model in the begining."""
        extract_model = self.trainer.model.cpu().state_dict()
        for key in extract_model.keys():
            self.weight_shapes[key] = extract_model[key].size()
            self.para_nums[key] = torch.numel(extract_model[key])

        self.encrypted_model = homo_enc.encrypt_weights(
            extract_model, True, self.ckks_context, []
        )

        # Store the initial model
        init_model_filename = f"{self.checkpoint_path}/{self.attack_prep_dir}/init.pth"
        with open(init_model_filename, "wb") as init_file:
            pickle.dump(self.encrypted_model["unencrypted_weights"], init_file)

        self.param_inited = True

    def mask_consensus(self, updates):
        """Conduct mask consensus on the reported mask proposals."""
        proposals = [update.payload for update in updates]
        mask_size = len(proposals[0])
        if mask_size == 0:
            self.final_mask = torch.tensor([])
        else:
            interleaved_indices = torch.zeros((sum([len(x) for x in proposals])))
            for i in range(len(proposals)):
                interleaved_indices[i :: len(proposals)] = proposals[i]

            _, indices = interleaved_indices.unique(sorted=False, return_inverse=True)

            self.final_mask = (
                interleaved_indices[indices.unique()[:mask_size]].clone().detach()
            )
            self.final_mask = self.final_mask.int().long()

    def _aggregate(self, updates):
        """Aggregate the model updates in the hybrid form of encrypted and unencrypted weights."""
        self.encrypted_model = self._fedavg_hybrid(updates)

        # Decrypt model weights for test accuracy
        decrypted_weights = homo_enc.decrypt_weights(
            self.encrypted_model, self.weight_shapes, self.para_nums
        )

        # Save the latest global model weights for evaluation
        latest_model_filename = (
            f"{self.checkpoint_path}/{self.attack_prep_dir}/latest.pth"
        )
        with open(latest_model_filename, "wb") as latest_file:
            pickle.dump(decrypted_weights, latest_file)

        self.encrypted_model["encrypted_weights"] = self.encrypted_model[
            "encrypted_weights"
        ].serialize()
        return decrypted_weights

    def _fedavg_hybrid(self, updates):
        """Aggregate the model updates in the hybrid form of encrypted and unencrypted weights."""
        weights_received = [
            homo_enc.deserialize_weights(update.payload, self.ckks_context)
            for update in updates
        ]

        unencrypted_weights = [x["unencrypted_weights"] for x in weights_received]
        encrypted_weights = [x["encrypted_weights"] for x in weights_received]

        # Extract the total number of samples
        self.total_samples = sum(update.report.num_samples for update in updates)

        # Perform weighted averaging
        unencrypted_avg_update = self.trainer.zeros(unencrypted_weights[0].size())
        encrypted_avg_update = self.trainer.zeros(encrypted_weights[0].size())

        for i, (unenc_w, enc_w) in enumerate(
            zip(unencrypted_weights, encrypted_weights)
        ):
            report = updates[i].report
            num_samples = report.num_samples

            unencrypted_avg_update += unenc_w * (num_samples / self.total_samples)
            encrypted_avg_update += enc_w * (num_samples / self.total_samples)

        # Wrap up the aggregation result
        avg_result = OrderedDict()
        avg_result["unencrypted_weights"] = unencrypted_avg_update
        avg_result["encrypted_weights"] = encrypted_avg_update
        avg_result["encrypt_indices"] = self.final_mask
        return avg_result
