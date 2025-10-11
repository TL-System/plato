"""
Customized NAS algorithms for PerFedRLNAS.
"""

import fedtools
import numpy as np
from model.mobilenetv3_supernet import NasDynamicModel

from plato.algorithms import fedavg
from plato.config import Config


class ServerAlgorithmSync(fedavg.Algorithm):
    """The federated learning algorithm for PerFedRLNAS, used by the server."""

    def __init__(self, trainer=None):
        super().__init__(trainer)
        self.current_subnet = None

    def extract_weights(self, model=None):
        payload = self.current_subnet.cpu().state_dict()
        return payload

    def load_weights(self, weights):
        pass

    def sample_config(self, server_response):
        """Sample ViT config to generate a subnet."""
        if (
            hasattr(Config().parameters.architect, "max_net")
            and Config().parameters.architect.max_net
        ):
            subnet_config = self.trainer.model.model.sample_max_subnet()
        else:
            subnet_config = self.trainer.model.sample_config(
                client_id=server_response["id"] - 1
            )
        subnet = fedtools.sample_subnet_w_config(self.model.model, subnet_config, True)
        self.current_subnet = subnet
        return subnet_config

    def nas_aggregation(
        self, subnets_config, weights_received, client_id_list, num_samples
    ):
        """Weight aggregation in NAS."""
        client_models = []
        subnet_configs = []
        for i, client_id_ in enumerate(client_id_list):
            client_id = client_id_ - 1
            subnet_config = subnets_config[client_id]
            client_model = fedtools.sample_subnet_w_config(
                self.model.model, subnet_config, False
            )
            client_model.load_state_dict(weights_received[i], strict=True)
            client_models.append(client_model)
            subnet_configs.append(subnet_config)
        neg_ratio = fedtools.fuse_weight(
            self.model.model,
            client_models,
            subnet_configs,
            num_samples,
        )
        return neg_ratio

    def set_active_subnet(self, cfg):
        """Set the suupernet to subnet with given cfg."""
        fedtools.set_active_subnet(self.model.model, cfg)

    def get_baseline_accuracy_info(self):
        """Get the information of accuracies of all clients."""
        accuracies = np.array([item[1] for item in self.model.baseline.items()])
        info = {
            "mean": np.mean(accuracies),
            "std": np.std(accuracies),
            "max": np.max(accuracies),
            "min": np.min(accuracies),
        }
        return info


class ServerAlgorithmAsync(ServerAlgorithmSync):
    """Server algorithm if asynchronous mode."""

    def nas_aggregation_async(
        self, aggregation_weight, subnets_config, weights_received, client_id_list
    ):
        """Weight aggregation in NAS."""
        client_models = []
        subnet_configs = []
        for i, client_id_ in enumerate(client_id_list):
            client_id = client_id_ - 1
            subnet_config = subnets_config[client_id]
            client_model = fedtools.sample_subnet_w_config(
                self.model.model, subnet_config, False
            )
            client_model.load_state_dict(weights_received[i], strict=True)
            client_models.append(client_model)
            subnet_configs.append(subnet_config)
        neg_ratio = fedtools.fuse_weight(
            self.model.model, client_models, subnet_configs, aggregation_weight
        )
        return neg_ratio


if hasattr(Config().server, "synchronous") and not Config().server.synchronous:
    ServerAlgorithm = ServerAlgorithmAsync
else:
    ServerAlgorithm = ServerAlgorithmSync


class ClientAlgorithm(fedavg.Algorithm):
    """The federated learning algorithm for PerFedRLNAS, used by the client."""

    def __init__(self, trainer=None):
        super().__init__(trainer)

    def extract_weights(self, model=None):
        if model is None:
            model = self.model
        return model.cpu().state_dict()

    def load_weights(self, weights):
        self.model.load_state_dict(weights, strict=True)

    def generate_client_model(self, subnet_config):
        """Generates the structure of the client model."""
        return fedtools.sample_subnet_w_config(NasDynamicModel(), subnet_config, False)
