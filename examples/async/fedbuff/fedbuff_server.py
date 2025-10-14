"""
A federated learning server using FedBuff.

Reference:

Nguyen, J., Malik, K., Zhan, H., et al., "Federated Learning with Buffered Asynchronous Aggregation,
" in Proc. International Conference on Artificial Intelligence and Statistics (AISTATS 2022).

https://proceedings.mlr.press/v151/nguyen22b/nguyen22b.pdf
"""

from plato.servers import fedavg
from plato.servers.strategies import FedBuffAggregationStrategy


class Server(fedavg.Server):
    """A federated learning server using the FedBuff aggregation strategy."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=FedBuffAggregationStrategy(),
        )
