"""
A federated learning server using Active Federated Learning with strategy pattern.

This is the updated version using the strategy-based API instead of inheritance.

Reference:

Goetz et al., "Active Federated Learning", 2019.

https://arxiv.org/pdf/1909.12641.pdf
"""

from plato.config import Config
from plato.servers import fedavg
from plato.servers.strategies import AFLSelectionStrategy


class Server(fedavg.Server):
    """A federated learning server using the AFL client selection strategy."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        # Load AFL parameters from config
        alpha1 = Config().algorithm.alpha1
        alpha2 = Config().algorithm.alpha2
        alpha3 = Config().algorithm.alpha3

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            client_selection_strategy=AFLSelectionStrategy(
                alpha1=alpha1,
                alpha2=alpha2,
                alpha3=alpha3,
            ),
        )
