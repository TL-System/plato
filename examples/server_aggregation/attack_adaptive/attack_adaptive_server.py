"""
Server aggregation using attack-adaptive aggregation.

Reference:

Ching Pui Wan, Qifeng Chen, "Robust Federated Learning with Attack-Adaptive Aggregation"
Unpublished
(https://arxiv.org/pdf/2102.05257.pdf)

Comparison to FedAtt, instead of using norm distance, this algorithm uses cosine
similarity between the client and server parameters. It also applies softmax with
temperatures.
"""

from collections import OrderedDict
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from attack_adaptive_server_strategy import AttackAdaptiveAggregationStrategy

from plato.config import Config
from plato.servers import fedavg
from plato.servers.strategies.base import AggregationStrategy, ServerContext


class Server(fedavg.Server):
    """
    A federated learning server using attack-adaptive aggregation strategy.

    The attack-adaptive aggregation logic is implemented in the aggregation strategy,
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
        # Use attack-adaptive aggregation strategy by default
        if aggregation_strategy is None:
            aggregation_strategy = AttackAdaptiveAggregationStrategy()

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=aggregation_strategy,
            client_selection_strategy=client_selection_strategy,
        )
