"""
The trainer for paper system-heterogenous federated learning through architecture search.
"""

from plato.config import Config
from plato.trainers import basic


class ServerTrainer(basic.Trainer):
    """A federated learning trainer of HeteroFL, used by the server."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model=model, callbacks=callbacks)
        self.model_class = model
        self.model = model(**Config().parameters.model._asdict())
        self.biggest_net_config = None
