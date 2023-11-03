"""
Implementation of Search Phase in Federared Model Search via Reinforcement Learning (FedRLNAS).

Reference:

Yao et al., "Federated Model Search via Reinforcement Learning", in the Proceedings of ICDCS 2021.

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9546522
"""
from types import SimpleNamespace

from plato.clients import simple


class Client(simple.Client):
    """A FedRLNAS client. Different clients hold different models."""

    def process_server_response(self, server_response) -> None:
        self.algorithm.mask_normal = server_response["mask_normal"]
        self.algorithm.mask_reduce = server_response["mask_reduce"]

        self.algorithm.model = self.algorithm.generate_client_model(
            self.algorithm.mask_normal, self.algorithm.mask_reduce
        )
        self.trainer.model = self.algorithm.model

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        """Mask information should be sent to the server for supernet aggregation."""
        report.mask_normal = self.algorithm.mask_normal
        report.mask_reduce = self.algorithm.mask_reduce

        return report
