"""
Federared Model Search via Reinforcement Learning

Reference:

Yao et al., "Federated Model Search via Reinforcement Learning", in the Proceedings of ICDCS 2021

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9546522
"""
import logging

from plato.config import Config
from plato.servers import fedavg

from Darts.model_search_local import MaskedNetwork
from fednas_tools import extract_index, fuse_weight_gradient, sample_mask


class Server(fedavg.Server):
    """FedNAS server, assign and aggregate model with diff arch"""

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        if (
            hasattr(Config().parameters.architect, "e_greedy")
            and Config().parameters.architect.e_greedy.epsilon > 0
            and Config().parameters.architect.e_greedy.epsilon < 1
        ):
            epsilon = Config().parameters.architect.e_greedy.epsilon
        else:
            epsilon = 0
        mask_normal = sample_mask(self.algorithm.model.alphas_normal, epsilon)
        mask_reduce = sample_mask(self.algorithm.model.alphas_reduce, epsilon)
        self.algorithm.mask_normal = mask_normal
        self.algorithm.mask_reduce = mask_reduce
        server_response["mask_normal"] = mask_normal
        server_response["mask_reduce"] = mask_reduce
        return server_response

    async def wrap_up(self):
        await super().wrap_up()
        logging.info("[%s] geneotypes: %s\n", self, self.trainer.model.model.genotype())

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """fuse weight of models with different arch"""
        masks_normal = [update.report.mask_normal for update in updates]
        masks_reduce = [update.report.mask_reduce for update in updates]

        # NAS aggregation
        client_models = []

        for i, payload in enumerate(weights_received):
            mask_normal = masks_normal[i]
            mask_reduce = masks_reduce[i]
            client_model = MaskedNetwork(
                Config().parameters.model.C,
                Config().parameters.model.num_classes,
                Config().parameters.model.layers,
                mask_normal,
                mask_reduce,
            )
            client_model.load_state_dict(payload, strict=True)
            client_models.append(client_model)
        fuse_weight_gradient(
            self.trainer.model.model,
            client_models,
            [update.report.num_samples for update in updates],
        )

    def weights_aggregated(self, updates):
        """after weight aggregation, update the arch parameter alpha"""
        accuracy_list = [update.report.accuracy for update in updates]
        mask_normals = [update.report.mask_normal for update in updates]
        mask_reduces = [update.report.mask_reduce for update in updates]
        epoch_index_normal = []
        epoch_index_reduce = []

        for i in range(len(updates)):
            mask_normal = mask_normals[i]
            mask_reduce = mask_reduces[i]
            index_normal = extract_index(mask_normal)
            index_reduce = extract_index(mask_reduce)
            epoch_index_normal.append(index_normal)
            epoch_index_reduce.append(index_reduce)

        self.trainer.model.step(accuracy_list, epoch_index_normal, epoch_index_reduce)
        self.algorithm.model = self.trainer.model
