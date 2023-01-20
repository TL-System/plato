"""
Implement new algorithm: personalized federarted NAS.
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
