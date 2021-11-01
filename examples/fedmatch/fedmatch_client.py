"""
A federated semi-supervised learning client using FedMatch, and data samples on devices are mostly unlabeled.
Reference:
Jeong et al., "Federated Semi-supervised learning with inter-client consistency & disjoint learning", in the Proceedings of ICLR 2021.
https://arxiv.org/pdf/2006.12097.pdf 
"""
import os
from dataclasses import dataclass
from plato.algorithms import fedavg
from plato.clients import simple
import torch
from plato.config import Config
import time


@dataclass
class Report(base.Report):
    """A client report containing the means and variances."""
    mean: float
    variance: float


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

    def load_data(self) -> None:
        """Generating data and loading them onto this client."""
        data_loading_start_time = time.perf_counter()

        super().load_data()

        # split training the dataset for supervised learning and unsupervised learning
        if Config().data.semi_supervised_learning:

            num_sup_train = self.num_train_examples * Config(
            ).data.ratio_s2u / (Config().data.ratio_s2u + 1)
            num_unsup_train = self.num_train_examples * (
                1 / (1 + Config().data.ratio_s2u))

            self.sup_train, self.unsup_train = torch.utils.data.random_split(
                self.trainset, [num_sup_train, num_unsup_train])

        self.data_loading_time = time.perf_counter() - data_loading_start_time

    async def train(self):
        """ Fedmatch clients use different number of local epochs. """

        report, weights = await super().train(
        )  # obtain update from local trainer,
        # loss in the trainer should be changed due to semi-supervised learning property

        # compute for mean, and variance that should be sent to server for further clustering
        #mean =
        #variance =

        # send them back to server
        return Report(report.num_samples, report.accuracy), weights

    def load_payload(self, server_payload):
        """ Load model weights and helpers from server payload onto this client. """

        if isinstance(server_payload, list):
            fedavg.load_weights(server_payload[0])
            self.helpers = server_payload[1:]  # download helpers from server
        else:
            fedavg.load_weights(server_payload)
            self.helpers = None
