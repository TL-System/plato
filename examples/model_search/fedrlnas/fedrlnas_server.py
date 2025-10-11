"""
Implementation  of Search Phase in Federared Model Search via Reinforcement Learning (FedRLNAS).

Reference:

Yao et al., "Federated Model Search via Reinforcement Learning", in the Proceedings of ICDCS 2021.

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9546522
"""

import logging

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """The FedRLNAS server assigns and aggregates global models with different architectures."""

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        if (
            hasattr(Config().parameters.architect, "e_greedy")
            and Config().parameters.architect.e_greedy.epsilon > 0
            and Config().parameters.architect.e_greedy.epsilon < 1
        ):
            epsilon = Config().parameters.architect.e_greedy.epsilon
        else:
            epsilon = 0

        mask_normal, mask_reduce = self.algorithm.sample_mask(epsilon)
        server_response["mask_normal"] = mask_normal
        server_response["mask_reduce"] = mask_reduce

        return server_response

    async def wrap_up(self) -> None:
        await super().wrap_up()

        logging.info("[%s] geneotypes: %s\n", self, self.trainer.model.genotype())

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
        mask_normals = [update.report.mask_normal for update in updates]
        mask_reduces = [update.report.mask_reduce for update in updates]

        epoch_index_normal = []
        epoch_index_reduce = []

        for i in range(len(updates)):
            mask_normal = mask_normals[i]
            mask_reduce = mask_reduces[i]
            index_normal = self.algorithm.extract_index(mask_normal)
            index_reduce = self.algorithm.extract_index(mask_reduce)
            epoch_index_normal.append(index_normal)
            epoch_index_reduce.append(index_reduce)

        self.algorithm.model.step(accuracy_list, epoch_index_normal, epoch_index_reduce)
        self.trainer.model = self.algorithm.model
