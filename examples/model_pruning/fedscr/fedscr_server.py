"""
A federated learning server using FedSCR. The server extracts the model updates from each client,
aggregates them and adds them to the global model from the previous round.
"""

import os
from typing import Dict, Optional, TYPE_CHECKING, cast

import numpy as np

from plato.algorithms import fedavg as fedavg_algorithm
from plato.config import Config
from plato.servers import fedavg

if TYPE_CHECKING:
    from .fedscr_trainer import Trainer as FedSCRTrainer


class Server(fedavg.Server):
    """A federated learning server using the FedSCR algorithm."""

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

        # Loss variances for each communication round used by the adaptive algorithm
        self.loss_variances: list[float] = []
        self.mean_variance: float = 0.0

        # Model divergences received from each client
        self.divs: Dict[int, float] = {}

        # Average weight received updates from each client
        self.avg_update: Dict[int, float] = {}

        clients_config = Config().clients
        update_threshold = float(getattr(clients_config, "update_threshold", 0.3))

        self.update_thresholds: Dict[str, float] = {
            str(client_id): update_threshold
            for client_id in range(1, self.total_clients + 1)
        }

        self.orig_threshold: float = update_threshold

        # Hyperparameters used for the adaptive algorithm
        self.delta1: float = float(getattr(clients_config, "delta1", 1.0))
        self.delta2: float = float(getattr(clients_config, "delta2", 1.0))
        self.delta3: float = float(getattr(clients_config, "delta3", 1.0))

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        """Wraps up generating the server response with any additional information."""
        trainer = cast(Optional["FedSCRTrainer"], self.trainer)
        if (
            trainer is not None
            and trainer.use_adaptive
            and self.current_round > 1
        ):
            self.calc_threshold()
            server_response["update_thresholds"] = self.update_thresholds
        return server_response

    def calc_threshold(self):
        """Calculates new update thresholds for each client."""
        for client_id, divergence in self.divs.items():
            sigmoid = (
                self.delta1 * divergence
                + self.delta2 * self.avg_update.get(client_id, 0.0)
                + self.delta3 * self.mean_variance
            )
            self.update_thresholds[str(client_id)] = float(
                (1 / (1 + (np.exp(-sigmoid)))) * self.orig_threshold
            )

    # pylint: disable=unused-argument
    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Aggregates the reported weight updates from the selected clients."""
        deltas = await self.aggregate_deltas(updates, weights_received)
        algorithm = cast(
            Optional[fedavg_algorithm.Algorithm],
            self.algorithm,
        )
        if algorithm is None:
            raise RuntimeError("FedSCR requires a FedAvg-style algorithm.")
        updated_weights = algorithm.update_weights(deltas)
        return updated_weights

    def weights_aggregated(self, updates):
        """Extracts required information from client reports after aggregating weights."""
        trainer = cast(Optional["FedSCRTrainer"], self.trainer)
        if trainer is not None and trainer.use_adaptive:
            # Compute mean of loss variances
            self.loss_variances.append(
                float(np.var([update.report.loss for update in updates]))
            )
            if self.current_round > 3:
                self.mean_variance = sum(self.loss_variances) / (
                    self.current_round - 2
                )
            else:
                self.mean_variance = 0.0

            self.divs = {
                update.client_id: update.report.div_from_global for update in updates
            }
            self.avg_update = {
                update.client_id: update.report.avg_update for update in updates
            }

    def server_will_close(self) -> None:
        """Method called at the start of closing the server."""
        model_name = Config().trainer.model_name
        checkpoint_path = Config().params["checkpoint_path"]

        # Delete files created by the clients.
        for client_id in range(1, self.total_clients + 1):
            acc_grad_file = f"{checkpoint_path}/{model_name}_client{client_id}_grad.pth"
            if os.path.exists(acc_grad_file):
                os.remove(acc_grad_file)
