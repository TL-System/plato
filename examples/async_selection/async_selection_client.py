from plato.clients import simple
from dataclasses import dataclass


class Client(simple.Client):
    """A personalized federated learning client using the FedRep algorithm."""
    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.stalenss = None
        self.local_gradient_norm = None

    async def train(self):
        """ Initialize the server control variate and client control variate for trainer. """
        report, weights = await super().train()

        return report, weights  # return stalenss and local gradient norm to server for next round sampling
