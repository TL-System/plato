import logging
import torch

from plato.config import Config
from plato.servers import fedavg

from Darts.model_search_local import MaskedNetwork
import torch.nn as nn
from fednas_tools import fuse_weight_gradient, extract_index, sample_mask


class Server(fedavg.Server):
    """Federated learning server using federated averaging."""

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(model, datasource, algorithm, trainer)

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        """Customizes the server response with any additional information."""
        mask_normal = sample_mask(
            self.algorithm.model.alphas_normal[server_response["id"] - 1]
        )
        mask_reduce = sample_mask(
            self.algorithm.model.alphas_reduce[server_response["id"] - 1]
        )
        self.algorithm.mask_normal = mask_normal
        self.algorithm.mask_reduce = mask_reduce
        server_response["mask_normal"] = mask_normal.numpy().tolist()
        server_response["mask_reduce"] = mask_reduce.numpy().tolist()
        self.algorithm.current_client_id = server_response["id"]
        return server_response

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        mask_normals = [update.report.mask_normal for update in updates]
        mask_reduces = [update.report.mask_reduce for update in updates]
        # NAS aggregation
        client_models = []

        for i, payload in enumerate(weights_received):
            mask_normal = torch.tensor(mask_normals[i])
            mask_reduce = torch.tensor(mask_reduces[i])
            client_model = MaskedNetwork(
                Config().parameters.model.C,
                Config().parameters.model.num_classes,
                Config().parameters.model.layers,
                nn.CrossEntropyLoss(),
                mask_normal,
                mask_reduce,
            )
            client_model.load_state_dict(payload, strict=True)
            client_models.append(client_model)
        if (
            hasattr(Config().parameters.architect, " personalize_last")
            and Config().parameters.architect.personalize_last
        ):
            fuse_weight_gradient(
                self.trainer.model.model,
                client_models,
                [update.report.num_samples for update in self.updates],
                False,
            )
        else:
            fuse_weight_gradient(
                self.trainer.model.model,
                client_models,
                [update.report.num_samples for update in self.updates],
            )

    def weights_aggregated(self, updates):
        accuracy_list = [update.report.accuracy for update in updates]
        mask_normals = [update.report.mask_normal for update in updates]
        mask_reduces = [update.report.mask_reduce for update in updates]
        client_id_list = [update.client_id for update in self.updates]
        epoch_index_normal = []
        epoch_index_reduce = []

        for i in range(len(updates)):
            mask_normal = torch.tensor(mask_normals[i])
            mask_reduce = torch.tensor(mask_reduces[i])
            index_normal = extract_index(mask_normal)
            index_reduce = extract_index(mask_reduce)
            epoch_index_normal.append(index_normal)
            epoch_index_reduce.append(index_reduce)

        if (
            hasattr(Config().parameters.architect, "warmup")
            and self.current_round < Config().parameters.architect.warmup
        ):
            pass
        else:
            # update alpha i s with value net
            if (
                hasattr(Config().parameters.architect, "natural_policy")
                and Config().parameters.architect.natural_policy
            ):
                self.trainer.model.step(
                    accuracy_list,
                    epoch_index_normal,
                    epoch_index_reduce,
                    client_id_list,
                )
            else:
                self.trainer.model.step(
                    accuracy_list,
                    epoch_index_normal,
                    epoch_index_reduce,
                    client_id_list,
                )

        self.algorithm.model = self.trainer.model

        # # # update value net, and use it to calculate baseline
        if hasattr(Config().parameters.architect, "value_net"):
            self.trainer.model.value_net.update(
                accuracy_list,
                client_id_list,
                self.trainer.model.alphas_normal,
                self.trainer.model.alphas_reduce,
            )
