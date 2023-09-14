"""
FedRolexFL algorithm trainer.
"""
from plato.config import Config
from plato.trainers.basic import Trainer


class ServerTrainer(Trainer):
    """A federated learning trainer of FedRolexFL, used by the server."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model=model, callbacks=callbacks)
        self.model = model(**Config().parameters.model._asdict())
