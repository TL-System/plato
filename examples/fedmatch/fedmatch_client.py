"""
A federated semi-supervised learning client using FedMatch, and data samples on devices
are mostly unlabeled.

Reference:

Jeong et al., "Federated Semi-supervised learning with inter-client consistency and
disjoint learning", in the Proceedings of ICLR 2021.

https://arxiv.org/pdf/2006.12097.pdf
"""
from plato.clients import simple
from plato.clients import base


class Client(simple.Client):
    """A fedmatch federated learning client who sends weight updates
    and the number of local epochs."""
    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.helpers = None
        self.sup_train = None
        self.unsup_train = None

    async def train(self):
        """ Fedmatch clients use different number of local epochs. """
        report, weights = await super().train(
        )  # obtain update from local trainer,
        # loss in the trainer should be changed due to semi-supervised learning property

        # compute for mean, and variance that should be sent to server for further clustering
        #mean =
        #variance =

        # send them back to server
        return base.Report(report.num_samples, report.accuracy), weights

    def load_payload(self, server_payload):
        """ Load model weights and helpers from server payload onto this client. """
        if isinstance(server_payload, list):
            self.algorithm.load_weights(server_payload[0])
            self.helpers = server_payload[1:]  # download helpers from server
        else:
            self.algorithm.load_weights(server_payload)
            self.helpers = None
