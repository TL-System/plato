"""
FedDiv Server

"""

import random
import torch
import numpy as np

from plato.servers import fedavg
from gmm_filter import GlobalFilterManager


class Server(fedavg.Server):
    """A MaskCrypt server with selective homomorphic encryption support."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(model, datasource, algorithm, trainer, callbacks)

        # Warm up
        self.warm_up = True
        self.warm_up_clients = []
        self.warm_up_iters = 0
        self.current_warm_up_round = None

        # Normal training
        self.global_filter = GlobalFilterManager(
            init_dataset=None, components=2, seed=None, init_params="random"
        )

    def choose_clients(self, clients_pool, clients_count):
        """Choose clients with no replacement in warm up phase."""
        if self.warm_up:
            if not len(self.warm_up_clients):
                self.warm_up_clients = clients_pool[:]
                random.shuffle(self.warm_up_clients)
                self.warm_up_iters += 1
            selected_clients = self.warm_up_clients[:clients_count]
            self.warm_up_clients = self.warm_up_clients[clients_count:]
            return selected_clients
        else:
            return super().choose_clients(clients_pool, clients_count)

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
            # Clients send model updates in even rounds, conduct aggregation
            aggregated_weights = await super().aggregate_weights(
                updates, baseline_weights, weights_received
            )
            return aggregated_weights
