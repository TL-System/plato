"""
Pisces: an asynchronous client selection and server aggregation algorithm.

Reference:

Z. Jiang, B. Wang, B. Li, B. Li. "Pisces: Efficient Federated Learning via Guided Asynchronous
Training," in Proceedings of ACM Symposium on Cloud Computing (SoCC), 2022.

URL: https://arxiv.org/abs/2206.09264
"""

from pisces_aggregation_strategy import PiscesAggregationStrategy
from pisces_selection_strategy import PiscesSelectionStrategy

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """Pisces server configured with aggregation and client selection strategies."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        server_cfg = Config().server

        staleness_factor = getattr(server_cfg, "staleness_factor", 1.0)
        history_window = getattr(server_cfg, "staleness_history_window", 5)

        aggregation_strategy = PiscesAggregationStrategy(
            staleness_factor=staleness_factor,
            history_window=history_window,
        )

        selection_strategy = PiscesSelectionStrategy(
            exploration_factor=getattr(server_cfg, "exploration_factor", 0.3),
            exploration_decaying_factor=getattr(
                server_cfg, "exploration_decaying_factor", 0.99
            ),
            min_explore_factor=getattr(server_cfg, "min_explore_factor", 0.1),
            staleness_factor=staleness_factor,
            robustness=getattr(server_cfg, "robustness", False),
            augmented_factor=getattr(server_cfg, "augmented_factor", 5),
            threshold_factor=getattr(server_cfg, "threshold_factor", 1.0),
            speed_penalty_factor=getattr(server_cfg, "speed_penalty_factor", 0.5),
            reliability_credit_initial=getattr(
                server_cfg, "reliability_credit_initial", 5
            ),
            history_window=history_window,
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
