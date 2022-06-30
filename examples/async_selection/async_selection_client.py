from plato.clients import simple


class Client(simple.Client):
    """A personalized federated learning client using the FedRep algorithm."""

    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)

    async def train(self):
        """ Initialize the server control variate and client control variate for trainer. """
        report, weights = await super().train()
        # calculate staleness

        # calculate local gradient norm

    

    return Report(report.num_samples, report.accuracy, 
                      report.training_time, comm_time, report.update_response,
                      2), [weights, self.client_control_variates]# return stalenss and local gradient norm to server for next round sampling

