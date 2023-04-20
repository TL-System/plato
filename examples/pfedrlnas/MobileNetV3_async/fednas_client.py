"""
Customized Client for PerFedRLNAS.
"""
from types import SimpleNamespace
from plato.config import Config

# pylint: disable=relative-beyond-top-level
from ..MobileNetV3.fednas_client import Client as sync_client


# pylint: disable=too-few-public-methods
class Client(sync_client):
    """A FedRLNAS client. Different clients hold different models."""

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        """Customize the information in report."""
        model_name = Config().trainer.model_name
        filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}.mem"
        max_mem_allocated, exceed_memory, sim_mem = self.trainer.load_memory(filename)
        if exceed_memory:
            report.accuracy = 0
        report.utilization = max_mem_allocated
        report.exceed = exceed_memory
        report.budget = sim_mem
        return super().customize_report(report)
