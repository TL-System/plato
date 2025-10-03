"""
Implement new algorithm: personalized federarted NAS.
"""

import logging
import pickle

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """The FedRLNAS server assigns and aggregates global models with different architectures."""

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        mask_normal, mask_reduce = self.algorithm.sample_mask(client_id)
        server_response["mask_normal"] = mask_normal
        server_response["mask_reduce"] = mask_reduce

        return server_response

    async def aggregate_weights(self, updates, baseline_weights, weights_received):  # pylint: disable=unused-argument
        """Aggregates weights of models with different architectures."""
        masks_normal = [update.report.mask_normal for update in updates]
        masks_reduce = [update.report.mask_reduce for update in updates]
        num_samples = [update.report.num_samples for update in updates]

        self.algorithm.nas_aggregation(
            masks_normal, masks_reduce, weights_received, num_samples
        )

    def weights_aggregated(self, updates):
        """After weight aggregation, update the architecture parameter alpha."""
        accuracy_list = [update.report.accuracy for update in updates]
        round_time_list = [
            update.report.training_time + update.report.comm_time
            for update in self.updates
        ]
        mask_normals = [update.report.mask_normal for update in updates]
        mask_reduces = [update.report.mask_reduce for update in updates]
        client_id_list = [update.client_id for update in self.updates]

        epoch_index_normal = []
        epoch_index_reduce = []

        for i in range(len(updates)):
            mask_normal = mask_normals[i]
            mask_reduce = mask_reduces[i]
            index_normal = self.algorithm.extract_index(mask_normal)
            index_reduce = self.algorithm.extract_index(mask_reduce)
            epoch_index_normal.append(index_normal)
            epoch_index_reduce.append(index_reduce)

        self.algorithm.model.step(
            [accuracy_list, round_time_list],
            epoch_index_normal,
            epoch_index_reduce,
            client_id_list,
        )
        self.trainer.model = self.algorithm.model

    def server_will_close(self):
        cfgs = []
        for i in range(1, Config().clients.total_clients + 1):
            cfg = self.algorithm.model.genotype(
                self.algorithm.model.alphas_normal[i - 1],
                self.algorithm.model.alphas_reduce[i - 1],
            )
            if cfg:
                logging.info("the config of client %s is %s", str(i), str(cfg))
                cfgs.append(cfg)
        # Use model_path if available, otherwise use default models/pretrained directory
        if hasattr(Config().server, "model_path"):
            model_dir = Config().server.model_path
        else:
            model_dir = "./models/pretrained"
        save_config = f"{model_dir}/subnet_configs.pickle"
        with open(save_config, "wb") as file:
            pickle.dump((cfgs), file)

    def save_to_checkpoint(self) -> None:
        # Similar way used in server_will_close
        cfgs = []
        for i in range(1, Config().clients.total_clients + 1):
            cfg = self.algorithm.model.genotype(
                self.algorithm.model.alphas_normal[i - 1],
                self.algorithm.model.alphas_reduce[i - 1],
            )
            if cfg:
                cfgs.append(cfg)

        # Use model_path if available, otherwise use default models/pretrained directory
        if hasattr(Config().server, "model_path"):
            model_dir = Config().server.model_path
        else:
            model_dir = "./models/pretrained"
        save_config = f"{model_dir}/subnet_configs.pickle"
        with open(save_config, "wb") as file:
            pickle.dump(cfgs, file)
        save_config = f"{model_dir}/baselines.pickle"
        with open(save_config, "wb") as file:
            pickle.dump(self.algorithm.model.baseline, file)
        return super().save_to_checkpoint()
