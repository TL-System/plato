"""
Implement new algorithm: personalized federated NAS.
"""

from __future__ import annotations

from typing import Protocol, cast

from Darts.model_search_local import MaskedNetwork
from fednas_tools import (
    client_weight_param,
    extract_index,
    fuse_weight_gradient,
    sample_mask,
)
from torch import Tensor
from torch.nn import Module

from plato.algorithms import fedavg
from plato.config import Config


class SupernetProtocol(Protocol):
    """Protocol describing the per-client supernet interface."""

    model: Module
    alphas_normal: list[Tensor]
    alphas_reduce: list[Tensor]
    baseline: dict

    def step(
        self,
        rewards_list,
        epoch_index_normal,
        epoch_index_reduce,
        client_id_list,
    ) -> None: ...

    def genotype(self, alphas_normal: Tensor, alphas_reduce: Tensor): ...


class FedNASAlgorithm(fedavg.Algorithm):
    """Basic algorithm for FedRLNAS."""

    def generate_client_model(self, mask_normal, mask_reduce):
        """Generates the structure of the client model."""
        client_model = MaskedNetwork(
            Config().parameters.model.C,
            Config().parameters.model.num_classes,
            Config().parameters.model.layers,
            mask_normal,
            mask_reduce,
        )

        return client_model


class ServerAlgorithm(FedNASAlgorithm):
    """The federated learning algorithm for FedRLNAS, used by the server."""

    def __init__(self, trainer=None):
        super().__init__(trainer)

        self.mask_normal: Tensor | None = None
        self.mask_reduce: Tensor | None = None

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
        """Extract weights from the supernet and assign different models to clients."""
        supernet = self._require_supernet()
        mask_normal = self.mask_normal
        mask_reduce = self.mask_reduce
        if mask_normal is None or mask_reduce is None:
            raise RuntimeError("Masks have not been sampled before extracting weights.")
        client_model = self.generate_client_model(mask_normal, mask_reduce)
        client_weight_param(supernet.model, client_model)
        return client_model.cpu().state_dict()

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""

    def sample_mask(self, client_id):
        """Sample mask to generate a subnet."""
        supernet = self._require_supernet()
        client_id -= 1
        mask_normal = sample_mask(supernet.alphas_normal[client_id])
        mask_reduce = sample_mask(supernet.alphas_reduce[client_id])
        self.mask_normal = mask_normal
        self.mask_reduce = mask_reduce
        return mask_normal, mask_reduce

    def nas_aggregation(
        self, masks_normal, masks_reduce, weights_received, num_samples
    ):
        """Weight aggregation in NAS."""
        supernet = self._require_supernet()
        client_models = []

        for i, payload in enumerate(weights_received):
            mask_normal = masks_normal[i]
            mask_reduce = masks_reduce[i]
            client_model = self.generate_client_model(mask_normal, mask_reduce)
            client_model.load_state_dict(payload, strict=True)
            client_models.append(client_model)
        fuse_weight_gradient(
            supernet.model,
            client_models,
            num_samples,
        )

    def extract_index(self, mask):
        """Extract edge index according to the mask."""
        return extract_index(mask)


class ClientAlgorithm(FedNASAlgorithm):
    """The federated learning algorithm for FedRLNAS, used by the client."""

    def __init__(self, trainer=None):
        super().__init__(trainer)

        self.mask_normal: Tensor | None = None
        self.mask_reduce: Tensor | None = None

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
