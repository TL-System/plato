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
from plato.servers.strategies import FedAsyncAggregationStrategy


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
        aggregation_strategy = FedAsyncAggregationStrategy()

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=aggregation_strategy,
        )

    def configure(self) -> None:
        """Configure the mixing hyperparameter for the server, as well as
        other parameters from the configuration file.
        """
        super().configure()

        if not hasattr(Config().server, "mixing_hyperparameter"):
            logging.warning(
                "FedAsync: Variable mixing hyperparameter is required for the FedAsync server."
            )
        else:
            try:
                mixing_hyperparam = float(Config().server.mixing_hyperparameter)
            except (TypeError, ValueError):
                logging.warning(
                    "FedAsync: Invalid mixing hyperparameter. "
                    "Unable to cast %s to float.",
                    Config().server.mixing_hyperparameter,
                )
                return

            if 0 < mixing_hyperparam < 1:
                logging.info(
                    "FedAsync: Mixing hyperparameter is set to %s.",
                    mixing_hyperparam,
                )
            else:
                logging.warning(
                    "FedAsync: Invalid mixing hyperparameter. "
                    "The hyperparameter needs to be between 0 and 1 (exclusive)."
                )
