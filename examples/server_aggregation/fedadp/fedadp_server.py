"""
Server aggregation using the FedADP strategy pattern.

Reference:

H. Wu, P. Wang. "Fast-Convergent Federated Learning with Adaptive Weighting," in IEEE Trans.
on Cognitive Communications and Networking (TCCN), 2021.

https://ieeexplore.ieee.org/abstract/document/9442814
"""

from fedadp_server_strategy import FedADPAggregationStrategy

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the FedADP aggregation strategy."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        aggregation_strategy=None,
        client_selection_strategy=None,
    ):
        if aggregation_strategy is None:
            alpha = (
                Config().algorithm.alpha if hasattr(Config().algorithm, "alpha") else 5
            )
            aggregation_strategy = FedADPAggregationStrategy(alpha=alpha)

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=aggregation_strategy,
            client_selection_strategy=client_selection_strategy,
        )
