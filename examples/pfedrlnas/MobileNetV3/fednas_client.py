"""
Customized Client for PerFedRLNAS.
"""
from types import SimpleNamespace

from plato.clients import simple
from plato.config import Config


class ClientSync(simple.Client):
    """A FedRLNAS client. Different clients hold different models."""

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(model, datasource, algorithm, trainer)

    def process_server_response(self, server_response) -> None:
        subnet_config = server_response["subnet_config"]
        self.algorithm.model = self.algorithm.generate_client_model(subnet_config)
        self.trainer.model = self.algorithm.model


class ClientAsync(ClientSync):
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


if hasattr(Config().server, "synchronous") and not Config().server.synchronous:
    Client = ClientAsync
else:
    Client = ClientSync
