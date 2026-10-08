"""
Customized NAS algorithms for PerFedRLNAS.
"""

from __future__ import annotations

from typing import Protocol, cast

import fedtools
import numpy as np
from nasvit_wrapper.attentive_nas_dynamic_model import (
    AttentiveNasDynamicModel,
)
from torch.nn import Module

from plato.algorithms import fedavg
from plato.config import Config


class SupernetProtocol(Protocol):
    """Protocol describing the Attentive NAS supernet interface."""

    model: Module
    baseline: dict

    def sample_max_subnet(self): ...

    def sample_config(self, client_id: int): ...

    def set_active_subnet(
        self, resolution, width, depth, kernel_size, expand_ratio
    ) -> None: ...

    def get_active_subnet(self, preserve_weight: bool = True) -> Module: ...

    def extract_index(self, subnets_config) -> list: ...

    def step(self, rewards_list, epoch_index, client_id_list) -> None: ...


class ServerAlgorithm(fedavg.Algorithm):
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

    def extract_weights(self, model=None):
        if self.current_subnet is None:
            raise RuntimeError("No subnet has been sampled yet for extraction.")
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
            supernet = self._require_supernet()
            base_model = cast(AttentiveNasDynamicModel, supernet.model)
            subnet_config = base_model.sample_max_subnet()
        else:
            supernet = self._require_supernet()
            subnet_config = supernet.sample_config(client_id=server_response["id"] - 1)
        base_model = cast(AttentiveNasDynamicModel, supernet.model)
        subnet = fedtools.sample_subnet_w_config(base_model, subnet_config, True)
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
            supernet = self._require_supernet()
            base_model = cast(AttentiveNasDynamicModel, supernet.model)
            client_model = fedtools.sample_subnet_w_config(
                base_model, subnet_config, False
            )
            client_model.load_state_dict(weights_received[i], strict=True)
            client_models.append(client_model)
            subnet_configs.append(subnet_config)
        supernet = self._require_supernet()
        base_model = cast(AttentiveNasDynamicModel, supernet.model)
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
        base_model = cast(AttentiveNasDynamicModel, supernet.model)
        fedtools.set_active_subnet(base_model, cfg)

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


class ClientAlgorithm(fedavg.Algorithm):
    """The federated learning algorithm for PerFedRLNAS, used by the client."""

    def __init__(self, trainer=None):
        super().__init__(trainer)

    def extract_weights(self, model=None):
        target_model: Module
        if model is not None:
            target_model = model
        else:
            target_model = cast(Module, self.require_model())
        return target_model.cpu().state_dict()

    def load_weights(self, weights):
        model = cast(Module, self.require_model())
        model.load_state_dict(weights, strict=True)

    def generate_client_model(self, subnet_config):
        """Generates the structure of the client model."""
        return fedtools.sample_subnet_w_config(
            AttentiveNasDynamicModel(), subnet_config, False
        )
