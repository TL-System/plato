"""
Implementation of Search Phase in Federated Model Search via Reinforcement Learning (FedRLNAS).

Reference:

Yao et al., "Federated Model Search via Reinforcement Learning", in the Proceedings of ICDCS 2021.
"""

from types import SimpleNamespace

from plato.clients import simple
from plato.clients.strategies import DefaultLifecycleStrategy


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
        reporting_strategy = self.reporting_strategy
        communication_strategy = self.communication_strategy

        self._configure_composable(
            lifecycle_strategy=FedRLNASLifecycleStrategy(),
            payload_strategy=payload_strategy,
            training_strategy=training_strategy,
            reporting_strategy=reporting_strategy,
            communication_strategy=communication_strategy,
        )

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        """Mask information should be sent to the server for supernet aggregation."""
        report.mask_normal = self.algorithm.mask_normal
        report.mask_reduce = self.algorithm.mask_reduce

        return report
