"""
A federated learning server using Active Federated Learning with the
strategy-based server API.

Clients are sampled according to valuation metrics computed on the client.

Reference:

Goetz et al., "Active Federated Learning", 2019.

https://arxiv.org/pdf/1909.12641.pdf
"""

from __future__ import annotations

from afl_selection_strategy import AFLSelectionStrategy

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """An AFL server configured with the strategy-based client selection API."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        algo_cfg = getattr(Config(), "algorithm", None)

        selection_strategy = AFLSelectionStrategy(
            alpha1=getattr(algo_cfg, "alpha1", 0.75) if algo_cfg else 0.75,
            alpha2=getattr(algo_cfg, "alpha2", 0.01) if algo_cfg else 0.01,
            alpha3=getattr(algo_cfg, "alpha3", 0.1) if algo_cfg else 0.1,
        )

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            client_selection_strategy=selection_strategy,
        )
