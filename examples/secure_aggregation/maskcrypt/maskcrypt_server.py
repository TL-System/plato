"""
A MaskCrypt server with selective homomorphic encryption support.
"""

from typing import cast

from maskcrypt_algorithm import Algorithm as MaskCryptAlgorithm

from plato.servers import fedavg_he


class Server(fedavg_he.Server):
    """A MaskCrypt server with selective homomorphic encryption support."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
    ):
        selected_algorithm = algorithm or MaskCryptAlgorithm
        super().__init__(model, datasource, selected_algorithm, trainer, callbacks)
        self.last_selected_clients = []

    def choose_clients(self, clients_pool, clients_count):
        """Choose the same clients every two rounds."""
        if self.current_round % 2 != 0 or not self.last_selected_clients:
            self.last_selected_clients = self._select_clients_with_strategy(
                clients_pool, clients_count
            )
        else:
            self.context.current_round = self.current_round
            self.client_selection_strategy.on_clients_selected(
                self.last_selected_clients, self.context
            )

        return self.last_selected_clients

    def customize_server_payload(self, payload):
        """Customize the server payload before sending to the client."""
        if self.current_round % 2 != 0:
            return self.encrypted_model
        else:
            return self.final_mask

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        if self.current_round % 2 != 0:
            # Clients send mask proposals in odd rounds, conduct mask consensus
            self._mask_consensus(updates)

            return baseline_weights
        else:
            strategy = getattr(self, "aggregation_strategy", None)
            if strategy is None or not hasattr(strategy, "aggregate_weights"):
                raise AttributeError(
                    "Aggregation strategy must expose an 'aggregate_weights' coroutine."
                )
            aggregated_weights = await strategy.aggregate_weights(
                updates, baseline_weights, weights_received, self.context
            )
            if aggregated_weights is None:
                raise RuntimeError("Aggregation strategy failed to produce weights.")

            return aggregated_weights

    def _mask_consensus(self, updates):
        """Conduct mask consensus on the reported mask proposals."""
        proposals = [update.payload for update in updates]
        algorithm = cast(MaskCryptAlgorithm, self.require_algorithm())
        self.final_mask = algorithm.build_consensus_mask(proposals)
