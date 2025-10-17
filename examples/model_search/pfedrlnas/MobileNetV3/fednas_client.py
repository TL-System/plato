"""
Customized Client for PerFedRLNAS.
"""

from plato.clients import simple
from plato.clients.strategies import DefaultLifecycleStrategy
from plato.clients.strategies.base import ClientContext
from plato.clients.strategies.defaults import DefaultReportingStrategy
from plato.config import Config


class PerFedRLNASMobileNetLifecycleStrategy(DefaultLifecycleStrategy):
    """Lifecycle strategy that applies server-provided subnet configurations."""

    _STATE_KEY = "pfedrlnas_mobilenet"

    @staticmethod
    def _state(context):
        return context.state.setdefault(
            PerFedRLNASMobileNetLifecycleStrategy._STATE_KEY, {}
        )

    def process_server_response(self, context, server_response) -> None:
        super().process_server_response(context, server_response)
        state = self._state(context)
        state["subnet_config"] = server_response.get("subnet_config")
        self._apply_if_ready(context, state)

    def configure(self, context) -> None:
        super().configure(context)
        state = self._state(context)
        self._apply_if_ready(context, state)

    def _apply_if_ready(self, context, state) -> None:
        subnet_config = state.get("subnet_config")
        if subnet_config is None:
            return

        algorithm = context.algorithm
        if algorithm is None:
            return

        model = algorithm.generate_client_model(subnet_config)
        algorithm.model = model

        trainer = context.trainer
        if trainer is not None:
            trainer.model = model

        owner = context.owner
        if owner is not None:
            if owner.algorithm is not None:
                owner.algorithm.model = model
            if owner.trainer is not None:
                owner.trainer.model = model


class PerFedRLNASMobileNetReportingStrategy(DefaultReportingStrategy):
    """Reporting strategy that optionally attaches memory utilisation metrics."""

    def __init__(self, *, include_memory_metrics: bool) -> None:
        super().__init__()
        self.include_memory_metrics = include_memory_metrics

    def build_report(self, context: ClientContext, report):
        report = super().build_report(context, report)
        if not self.include_memory_metrics:
            return report

        trainer = context.trainer
        if trainer is None:
            return report

        model_name = Config().trainer.model_name
        filename = f"{model_name}_{context.client_id}_{Config().params['run_id']}.mem"
        max_mem_allocated, exceed_memory, sim_mem = trainer.load_memory(filename)
        if exceed_memory:
            report.accuracy = 0
        report.utilization = max_mem_allocated
        report.exceed = exceed_memory
        report.budget = sim_mem
        return report


class ClientSync(simple.Client):
    """A FedRLNAS client. Different clients hold different models."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        trainer_callbacks=None,
        *,
        include_memory_metrics: bool = False,
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            trainer_callbacks=trainer_callbacks,
        )

        payload_strategy = self.payload_strategy
        training_strategy = self.training_strategy
        communication_strategy = self.communication_strategy

        self._configure_composable(
            lifecycle_strategy=PerFedRLNASMobileNetLifecycleStrategy(),
            payload_strategy=payload_strategy,
            training_strategy=training_strategy,
            reporting_strategy=PerFedRLNASMobileNetReportingStrategy(
                include_memory_metrics=include_memory_metrics
            ),
            communication_strategy=communication_strategy,
        )


class ClientAsync(ClientSync):
    """A FedRLNAS client. Different clients hold different models."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        trainer_callbacks=None,
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            trainer_callbacks=trainer_callbacks,
            include_memory_metrics=True,
        )


if hasattr(Config().server, "synchronous") and not Config().server.synchronous:
    Client = ClientAsync
else:
    Client = ClientSync
