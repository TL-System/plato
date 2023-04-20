"""
Customized NAS algorithms for PerFedRLNAS.
"""
# pylint: disable=relative-beyond-top-level
from ..MobileNetV3.fedtools import sample_subnet_w_config, fuse_weight
from ..MobileNetV3.fednas_algorithm import ServerAlgorithm as sync_server_algorithm


# pylint: disable=too-few-public-methods
class ServerAlgorithm(sync_server_algorithm):
    """The federated learning algorithm for PerFedRLNAS, used by the server."""

    def nas_aggregation(
        self, aggregation_weight, subnets_config, weights_received, client_id_list
    ):
        """Weight aggregation in NAS."""
        client_models = []
        subnet_configs = []
        for i, client_id_ in enumerate(client_id_list):
            client_id = client_id_ - 1
            subnet_config = subnets_config[client_id]
            client_model = sample_subnet_w_config(
                self.model.model, subnet_config, False
            )
            client_model.load_state_dict(weights_received[i], strict=True)
            client_models.append(client_model)
            subnet_configs.append(subnet_config)
        neg_ratio = fuse_weight(
            self.model.model, client_models, subnet_configs, aggregation_weight
        )
        return neg_ratio
