"""
Server aggregation using FedAtt.

Reference:

S. Ji, S. Pan, G. Long, X. Li, J. Jiang, Z. Huang. "Learning Private Neural Language Modeling
with Attentive Aggregation," in Proc. International Joint Conference on Neural Networks (IJCNN),
2019.

https://arxiv.org/abs/1812.07108
"""

from fedatt_algorithm import Algorithm as FedAttAlgorithm
from fedatt_server_strategy import FedAttAggregationStrategy

from plato.servers import fedavg


class Server(fedavg.Server):
    """
    A federated learning server using FedAtt aggregation strategy.

    The FedAtt aggregation logic is implemented in the aggregation strategy,
    following the composition-over-inheritance pattern.
    """

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
        # Use FedAtt aggregation strategy by default
        if aggregation_strategy is None:
            aggregation_strategy = FedAttAggregationStrategy()

        selected_algorithm = algorithm or FedAttAlgorithm

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=selected_algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=aggregation_strategy,
            client_selection_strategy=client_selection_strategy,
        )
