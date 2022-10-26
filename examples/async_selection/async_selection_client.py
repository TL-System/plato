from plato.clients import simple


class Client(simple.Client):
    """A personalized federated learning client using the FedRep algorithm."""

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.stalenss = None
        self.local_gradient_norm = None
        # set special trainer for test model; import a trainer class here for the first round training
        self.trainer_for_test_model = 

    async def train(self):
        """Initialize the server control variate and client control variate for trainer."""
        report, weights = await super().train()

        return (
            report,
            weights,
        )  # return stalenss and local gradient norm to server for next round sampling

    async def inbound_processed(self, processed_inbound_payload):
        """
        Override this method to conduct customized operations to generate a client's response to
        the server when inbound payload from the server has been processed.
        """
        # first round, clients train test model.
        # if current_round == 1: 
        # self.trainer_for_test_model.train()


        # for following round, clients train target model.
        report, outbound_payload = await self.start_training(processed_inbound_payload)
        return report, outbound_payload
