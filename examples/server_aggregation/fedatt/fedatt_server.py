"""
Server aggregation using FedAtt.

Reference:

S. Ji, S. Pan, G. Long, X. Li, J. Jiang, Z. Huang. "Learning Private Neural Language Modeling
with Attentive Aggregation," in Proc. International Joint Conference on Neural Networks (IJCNN),
2019.

https://arxiv.org/abs/1812.07108
"""

from collections import OrderedDict
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from fedatt_server_strategy import FedAttAggregationStrategy

from plato.config import Config
from plato.servers import fedavg
from plato.servers.strategies.base import AggregationStrategy, ServerContext


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

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=aggregation_strategy,
            client_selection_strategy=client_selection_strategy,
        )
