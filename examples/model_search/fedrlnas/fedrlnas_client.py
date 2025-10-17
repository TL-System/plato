"""
Implementation of Search Phase in Federated Model Search via Reinforcement Learning (FedRLNAS).

Reference:

Yao et al., "Federated Model Search via Reinforcement Learning", in the Proceedings of ICDCS 2021.
"""

from plato.clients import simple
from plato.clients.strategies import DefaultLifecycleStrategy
from plato.clients.strategies.base import ClientContext
from plato.clients.strategies.defaults import DefaultReportingStrategy


class FedRLNASLifecycleStrategy(DefaultLifecycleStrategy):
    """Lifecycle strategy that keeps client masks and generated models in sync."""

    _STATE_KEY = "fedrlnas"

    @staticmethod
    def _state(context):
        return context.state.setdefault(FedRLNASLifecycleStrategy._STATE_KEY, {})

    def process_server_response(self, context, server_response) -> None:
        super().process_server_response(context, server_response)
        state = self._state(context)
        state["mask_normal"] = server_response.get("mask_normal")
        state["mask_reduce"] = server_response.get("mask_reduce")
        self._apply_if_ready(context, state)

    def configure(self, context) -> None:
        super().configure(context)
        state = self._state(context)
        self._apply_if_ready(context, state)

    def _apply_if_ready(self, context, state) -> None:
        mask_normal = state.get("mask_normal")
        mask_reduce = state.get("mask_reduce")

        if mask_normal is None or mask_reduce is None:
            return

        algorithm = context.algorithm
        if algorithm is None:
            return

        algorithm.mask_normal = mask_normal
        algorithm.mask_reduce = mask_reduce
        model = algorithm.generate_client_model(mask_normal, mask_reduce)
        algorithm.model = model

        trainer = context.trainer
        if trainer is not None:
            trainer.model = model

        owner = context.owner
        if owner is not None:
            if owner.algorithm is not None:
                owner.algorithm.mask_normal = mask_normal
                owner.algorithm.mask_reduce = mask_reduce
                owner.algorithm.model = model
            if owner.trainer is not None:
                owner.trainer.model = model


class FedRLNASReportingStrategy(DefaultReportingStrategy):
    """Reporting strategy attaching mask metadata for aggregation."""

    def build_report(self, context: ClientContext, report):
        report = super().build_report(context, report)

        algorithm = context.algorithm
        if algorithm is None:
            return report

        report.mask_normal = getattr(algorithm, "mask_normal", None)
        report.mask_reduce = getattr(algorithm, "mask_reduce", None)
        return report


class Client(simple.Client):
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
        )

        payload_strategy = self.payload_strategy
        training_strategy = self.training_strategy
        communication_strategy = self.communication_strategy

        self._configure_composable(
            lifecycle_strategy=FedRLNASLifecycleStrategy(),
            payload_strategy=payload_strategy,
            training_strategy=training_strategy,
            reporting_strategy=FedRLNASReportingStrategy(),
            communication_strategy=communication_strategy,
        )
