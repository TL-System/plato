"""FedAvg server wrapper enabling MPC additive secret sharing."""

from __future__ import annotations

from plato.config import Config
from plato.mpc import RoundInfoStore
from plato.servers import fedavg
from plato.servers.strategies.mpc import (
    MPCAdditiveAggregationStrategy,
    MPCRoundSelectionStrategy,
)


class Server(fedavg.Server):
    """FedAvg server that reconstructs additive MPC shares before aggregation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        debug_flag = getattr(Config().clients, "mpc_debug_artifacts", False)
        self.round_store = RoundInfoStore.from_config(lock=self._mpc_round_lock)
        self.client_selection_strategy = MPCRoundSelectionStrategy(self.round_store)
        self.aggregation_strategy = MPCAdditiveAggregationStrategy(
            self.round_store, debug_flag
        )

    def configure(self) -> None:
        super().configure()
        self.round_store.reset()
        self.context.state["mpc_round_store"] = self.round_store
