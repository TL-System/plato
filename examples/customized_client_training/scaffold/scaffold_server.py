"""SCAFFOLD population controls with inherited sample-weighted model averaging.

With unequal sample counts this is a SCAFFOLD-derived extension, not the paper's
uniform-client model update or a proved weighted objective. Control updates
retain Algorithm 1's population denominator N.
Reference: https://arxiv.org/pdf/1910.06378v4.
"""

from collections import OrderedDict

import torch

from plato.config import Config
from plato.servers import fedavg
from plato.trainers.strategies.algorithms.scaffold_strategy import (
    validate_control_variates,
)


class Server(fedavg.Server):
    """Validate all participating deltas before model aggregation can commit."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )
        self.server_control_variate = None
        self.received_client_control_variates = None
        self._pending_control_variate = None

    def _model(self):
        if self.trainer is None or self.trainer.model is None:
            raise RuntimeError("SCAFFOLD server requires a trainer model.")
        return self.trainer.model

    def _controls(self):
        model = self._model()
        controls = self.server_control_variate
        if controls is None:
            controls = OrderedDict(
                (name, torch.zeros_like(parameter, device="cpu"))
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
        return validate_control_variates(model, controls)

    def weights_received(self, weights_received):
        """Stage c+sum(delta_ci)/N, rejecting malformed deltas before mutation."""
        self._pending_control_variate = None
        self.received_client_control_variates = None
        controls = self._controls()
        population = Config().clients.total_clients
        if not isinstance(population, int) or population <= 0:
            raise ValueError("SCAFFOLD requires a positive total client population.")
        deltas, weights = [], []
        for payload in weights_received:
            if not isinstance(payload, (list, tuple)) or len(payload) != 2:
                raise ValueError(
                    "SCAFFOLD requires every participating [weights, delta_ci] payload."
                )
            deltas.append(validate_control_variates(self._model(), payload[1]))
            weights.append(payload[0])
        if not deltas:
            raise ValueError("SCAFFOLD cannot aggregate an empty participating set.")
        staged = OrderedDict(
            (name, value + sum(delta[name] for delta in deltas) / population)
            for name, value in controls.items()
        )
        self._pending_control_variate = validate_control_variates(self._model(), staged)
        self.received_client_control_variates = deltas
        return weights

    def weights_aggregated(self, updates):
        """Commit the already validated population control update after weights."""
        if self._pending_control_variate is None:
            raise RuntimeError(
                "SCAFFOLD aggregation has no validated current-round controls."
            )
        self.server_control_variate = self._pending_control_variate
        self._pending_control_variate = None

    def customize_server_payload(self, payload):
        """Send independent trainable-parameter controls with the full model state."""
        self.server_control_variate = self._controls()
        return [
            payload,
            OrderedDict(
                (name, value.clone())
                for name, value in self.server_control_variate.items()
            ),
        ]
