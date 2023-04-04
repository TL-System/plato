"""
Customized Client for PerFedRLNAS.
"""
import logging
import pickle
from types import SimpleNamespace
from plato.clients import simple
from plato.config import Config


class Client(simple.Client):
    """A FedRLNAS client. Different clients hold different models."""

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(model, datasource, algorithm, trainer)

    def process_server_response(self, server_response) -> None:
        subnet_config = server_response["subnet_config"]
        self.trainer.current_config = subnet_config
        self.algorithm.model = self.algorithm.generate_client_model(subnet_config)
        self.trainer.model = self.algorithm.model

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        model_name = Config().trainer.model_name
        filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}.mem"
        max_mem_allocated, exceed_memory, sim_mem = self.trainer.load_memory(filename)
        if exceed_memory:
            report.accuracy = 0
        report.utilization = max_mem_allocated
        report.exceed = exceed_memory
        report.budget = sim_mem
        return super().customize_report(report)

    async def _request_update(self, data) -> None:
        """Upon receiving a request for an urgent model update."""
        logging.info(
            "[Client #%s] Urgent request received for model update at time %s.",
            data["client_id"],
            data["time"],
        )

        try:
            report, payload = await self._obtain_model_update(
                client_id=data["client_id"],
                requested_time=data["time"],
            )
        except ValueError:
            logging.info(
                f"[Client #{data['client_id']}] Cannot find an epoch that matches the wall-clock time provided. No return model."
            )
            return

        # Process outbound data when necessary
        self.callback_handler.call_event(
            "on_outbound_ready", self, report, self.outbound_processor
        )
        self.outbound_ready(report, self.outbound_processor)
        payload = self.outbound_processor.process(payload)

        # Sending the client report as metadata to the server (payload to follow)
        await self.sio.emit(
            "client_report", {"id": self.client_id, "report": pickle.dumps(report)}
        )

        # Sending the client training payload to the server
        await self._send(payload)
