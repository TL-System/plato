from plato.clients import simple
from dataclasses import dataclass


@dataclass
class Report(simple.Report):
    """Client report sent to the FedSarah federated learning server."""
    local_gradient_norm: float


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

        # calculate local gradient norm

        return Report(
            report.num_samples, report.accuracy, report.training_time,
            report.comm_time, report.update_response, self.local_gradient_norm
        ), weights  # return stalenss and local gradient norm to server for next round sampling
