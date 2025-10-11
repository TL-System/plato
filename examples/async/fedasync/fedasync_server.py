"""
A federated learning server using FedAsync.

Reference:

Xie, C., Koyejo, S., Gupta, I. "Asynchronous federated optimization,"
in Proc. 12th Annual Workshop on Optimization for Machine Learning (OPT 2020).

https://opt-ml.org/papers/2020/paper_28.pdf
"""

import logging

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the FedAsync algorithm."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        # The hyperparameter of FedAsync with a range of (0, 1)
        self.mixing_hyperparam = 1

        # Whether adjust mixing hyperparameter after each round
        self.adaptive_mixing = False

    def configure(self) -> None:
        """Configure the mixing hyperparameter for the server, as well as
        other parameters from the configuration file.
        """
        super().configure()

        # Configuring the mixing hyperparameter for FedAsync
        self.adaptive_mixing = (
            hasattr(Config().server, "adaptive_mixing")
            and Config().server.adaptive_mixing
        )

        if not hasattr(Config().server, "mixing_hyperparameter"):
            logging.warning(
                "FedAsync: Variable mixing hyperparameter is required for the FedAsync server."
            )
        else:
            self.mixing_hyperparam = Config().server.mixing_hyperparameter

            if 0 < self.mixing_hyperparam < 1:
                logging.info(
                    "FedAsync: Mixing hyperparameter is set to %s.",
                    self.mixing_hyperparam,
                )
            else:
                logging.warning(
                    "FedAsync: Invalid mixing hyperparameter. "
                    "The hyperparameter needs to be between 0 and 1 (exclusive)."
                )

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Process the client reports by aggregating their weights."""
        # Calculate the new mixing hyperparameter with client's staleness
        client_staleness = updates[0].staleness

        if self.adaptive_mixing:
            self.mixing_hyperparam *= self._staleness_function(client_staleness)

        return await self.algorithm.aggregate_weights(
            baseline_weights, weights_received, mixing=self.mixing_hyperparam
        )

    @staticmethod
    def _staleness_function(staleness) -> float:
        """Staleness function used to adjust the mixing hyperparameter"""
        if hasattr(Config().server, "staleness_weighting_function"):
            staleness_func_param = Config().server.staleness_weighting_function
            func_type = staleness_func_param.type.lower()
            if func_type == "constant":
                return Server._constant_function()
            elif func_type == "polynomial":
                a = staleness_func_param.a
                return Server._polynomial_function(staleness, a)
            elif func_type == "hinge":
                a = staleness_func_param.a
                b = staleness_func_param.b
                return Server._hinge_function(staleness, a, b)
            else:
                logging.warning(
                    "FedAsync: Unknown staleness weighting function type. "
                    "Type needs to be constant, polynomial, or hinge."
                )
        else:
            return Server.constant_function()

    @staticmethod
    def _constant_function() -> float:
        """Constant staleness function as proposed in Sec. 5.2, Evaluation Setup."""
        return 1

    @staticmethod
    def _polynomial_function(staleness, a) -> float:
        """Polynomial staleness function as proposed in Sec. 5.2, Evaluation Setup."""
        return (staleness + 1) ** -a

    @staticmethod
    def _hinge_function(staleness, a, b) -> float:
        """Hinge staleness function as proposed in Sec. 5.2, Evaluation Setup."""
        if staleness <= b:
            return 1
        else:
            return 1 / (a * (staleness - b) + 1)
