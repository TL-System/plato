"""
A federated learning server using FedNova with strategy pattern.

This is the updated version using the strategy-based API instead of inheritance.

Reference:

Wang et al., "Tackling the Objective Inconsistency Problem in Heterogeneous Federated
Optimization", in the Proceedings of NeurIPS 2020.

https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html
"""

from plato.servers import fedavg
from plato.servers.strategies import FedNovaAggregationStrategy


class Server(fedavg.Server):
    """A federated learning server using the FedNova aggregation strategy."""

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
            aggregation_strategy=FedNovaAggregationStrategy(),
        )
