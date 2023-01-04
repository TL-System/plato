"""
A simple federated learning server capable of selecting clients unseen during
training.

The total clients are divided into two parts, referred to as
1.- participant clients
2.- unparticipant clients

"""

from plato.servers import fedavg
from plato.config import Config


class Server(fedavg.Server):
    """Federated learning server controling the client selection."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        # two types of client groups
        # utilized clients during federated training
        self.participant_clients = 0
        # unused clients during federated training
        self.nonparticipant_clients = 0

        # clients id of two types of clients
        self.participant_clients_pool = []
        self.nonparticipant_clients_pool = []

        self.load_clients_group()

    def load_clients_group(self):
        """Loaded two types of clients."""
        # set participant and nonparticipant clients
        # by default,
        #  total clients will participant in federated training
        self.participant_clients = (
            Config().clients.participant_clients
            if hasattr(Config().clients, "participant_clients")
            else self.total_clients
        )
        self.nonparticipant_clients = (
            Config().clients.participant_clients
            if hasattr(Config().clients, "participant_clients")
            else 0
        )

        self.participant_clients_pool = (
            Config().clients.participant_clients_id
            if hasattr(Config().clients, "participant_clients_id")
            else []
        )

        self.nonparticipant_clients_pool = (
            Config().clients.nonparticipant_clients_pool
            if hasattr(Config().clients, "nonparticipant_clients_id")
            else []
        )

    def initialize_clients_pool(self):
        """Initial clients pool to be selected for federated learning."""

        # the default clients pool
        super().initialize_clients_pool()

        # set the clients pool by clients groups
        if self.participant_clients_pool:
            self.clients_pool = self.participant_clients_pool

        self.nonparticipant_clients_pool = [
            client_id
            for client_id in self.clients_pool
            if client_id not in self.participant_clients_pool
        ]
