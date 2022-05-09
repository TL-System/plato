"""
A customized server with asynchronous client selection
"""
from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)

    def choose_clients(self, clients_pool, clients_count):
        return super().choose_clients(clients_pool, clients_count)
