"""
Polaris: asynchronous client selection via geometric programming.

Reference:

L. Yang, W. Wang, and V. Smith, "Polaris: Efficient Federated Learning via
Intelligent Client Selection," 2020.
"""

from examples.client_selection.polaris.polaris_aggregation_strategy import (
    PolarisAggregationStrategy,
)
from examples.client_selection.polaris.polaris_selection_strategy import (
    PolarisSelectionStrategy,
)
from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """Polaris server configured with aggregation and selection strategies."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        server_cfg = Config().server

        aggregation_strategy = PolarisAggregationStrategy(
            alpha=getattr(server_cfg, "polaris_alpha", 10.0),
            initial_gradient_bound=getattr(
                server_cfg, "polaris_initial_gradient_bound", 0.5
            ),
            initial_staleness=getattr(server_cfg, "polaris_initial_staleness", 0.01),
        )

        selection_strategy = PolarisSelectionStrategy(
            beta=getattr(server_cfg, "polaris_beta", 1.0),
            staleness_weight=getattr(server_cfg, "polaris_staleness_weight", 1.0),
        )

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=aggregation_strategy,
            client_selection_strategy=selection_strategy,
        )
