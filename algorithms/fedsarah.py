from algorithms import fedavg
from trainers.fedsarah import Trainer


class Algorithm(fedavg.Algorithm):
    """The FedSarah algorithm, used by both the client and the
    server.
    """
    def __init__(self, trainer: Trainer, client_id=None):
        """
        """
        self.trainer = trainer
        self.model = trainer.model
        self.client_id = client_id
