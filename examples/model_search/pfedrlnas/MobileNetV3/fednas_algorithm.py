"""
Customized NAS algorithms for PerFedRLNAS.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

import fedtools
import numpy as np
from model.mobilenetv3_supernet import NasDynamicModel
from torch.nn import Module

from plato.algorithms import fedavg
from plato.config import Config


class SupernetProtocol(Protocol):
    """Protocol describing the MobileNetV3 supernet interface."""

    model: NasDynamicModel
    baseline: dict[Any, Any]

    def sample_max_subnet(self) -> dict[str, Any]: ...

    def sample_config(self, client_id: int) -> dict[str, Any]: ...

    def set_active_subnet(
        self, resolution, width, depth, kernel_size, expand_ratio
    ) -> None: ...

    def get_active_subnet(self, preserve_weight: bool = True) -> Module: ...

    def extract_index(self, subnets_config) -> list[Any]: ...

    def step(
        self,
        rewards_list,
        epoch_index,
        client_id_list,
    ) -> None: ...


class ServerAlgorithmSync(fedavg.Algorithm):
    """The federated learning algorithm for PerFedRLNAS, used by the server."""

    def __init__(self, trainer=None):
        super().__init__(trainer)
        self.current_subnet: Module | None = None

    def _require_supernet(self) -> SupernetProtocol:
        if self.model is not None:
            return cast(SupernetProtocol, self.model)
        trainer = self.require_trainer()
        supernet = getattr(trainer, "model", None)
        if supernet is None:
            raise RuntimeError("Supernet model is not attached to the trainer.")
        self.model = supernet
        return cast(SupernetProtocol, supernet)

    def get_supernet(self) -> SupernetProtocol:
        """Expose the supernet reference for collaborators such as the server."""
        return self._require_supernet()

    def extract_weights(self, model=None):
        if self.current_subnet is None:
            raise RuntimeError("No subnet has been sampled yet for extraction.")
        payload = self.current_subnet.cpu().state_dict()
        return payload

    def load_weights(self, weights):
        pass

    def sample_config(self, server_response):
        """Sample ViT config to generate a subnet."""
        supernet = self._require_supernet()
        base_model = supernet.model
        if (
            hasattr(Config().parameters.architect, "max_net")
            and Config().parameters.architect.max_net
        ):
            subnet_config = base_model.sample_max_subnet()
        else:
            subnet_config = supernet.sample_config(client_id=server_response["id"] - 1)
        subnet = fedtools.sample_subnet_w_config(base_model, subnet_config, True)
        self.current_subnet = subnet
        return subnet_config

    def nas_aggregation(
        self, subnets_config, weights_received, client_id_list, num_samples
    ):
        """Weight aggregation in NAS."""
        supernet = self._require_supernet()
        base_model = supernet.model
        client_models = []
        subnet_configs = []
        for i, client_id_ in enumerate(client_id_list):
            client_id = client_id_ - 1
            subnet_config = subnets_config[client_id]
            client_model = fedtools.sample_subnet_w_config(
                base_model, subnet_config, False
            )
            client_model.load_state_dict(weights_received[i], strict=True)
            client_models.append(client_model)
            subnet_configs.append(subnet_config)
        neg_ratio = fedtools.fuse_weight(
            base_model,
            client_models,
            subnet_configs,
            num_samples,
        )
        return neg_ratio

    def set_active_subnet(self, cfg):
        """Set the suupernet to subnet with given cfg."""
        supernet = self._require_supernet()
        fedtools.set_active_subnet(supernet.model, cfg)

    def get_baseline_accuracy_info(self):
        """Get the information of accuracies of all clients."""
        supernet = self._require_supernet()
        accuracies = np.array([item[1] for item in supernet.baseline.items()])
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
        supernet = self._require_supernet()
        base_model = supernet.model
        for i, client_id_ in enumerate(client_id_list):
            client_id = client_id_ - 1
            subnet_config = subnets_config[client_id]
            client_model = fedtools.sample_subnet_w_config(
                base_model, subnet_config, False
            )
            client_model.load_state_dict(weights_received[i], strict=True)
            client_models.append(client_model)
            subnet_configs.append(subnet_config)
        neg_ratio = fedtools.fuse_weight(
            base_model, client_models, subnet_configs, aggregation_weight
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
        target_model: Module
        if model is None:
            target_model = cast(Module, self.require_model())
        else:
            target_model = model
        return target_model.cpu().state_dict()

    def load_weights(self, weights):
        model = cast(Module, self.require_model())
        model.load_state_dict(weights, strict=True)

    def generate_client_model(self, subnet_config):
        """Generates the structure of the client model."""
        return fedtools.sample_subnet_w_config(NasDynamicModel(), subnet_config, False)
