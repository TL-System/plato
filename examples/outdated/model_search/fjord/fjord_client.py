"""
Customized Client for FjORD.
"""

from plato.clients import simple
from plato.clients.strategies import DefaultLifecycleStrategy
from plato.config import Config


class FjordLifecycleStrategy(DefaultLifecycleStrategy):
    """Lifecycle strategy that adapts the client model to server-provided rates."""

    _STATE_KEY = "fjord"

    @staticmethod
    def _state(context):
        return context.state.setdefault(FjordLifecycleStrategy._STATE_KEY, {})

    def process_server_response(self, context, server_response) -> None:
        super().process_server_response(context, server_response)
        state = self._state(context)
        state["rate"] = server_response.get("rate")
        self._apply_if_ready(context, state)

    def configure(self, context) -> None:
        super().configure(context)
        state = self._state(context)
        self._apply_if_ready(context, state)

    def _apply_if_ready(self, context, state) -> None:
        rate = state.get("rate")
        if rate is None:
            return

        model_factory = context.custom_model or context.model
        if model_factory is None:
            return

        model = model_factory(
            model_rate=rate,
            **Config().parameters.client_model._asdict(),
        )

        trainer = context.trainer
        if trainer is not None:
            trainer.rate = rate
            trainer.model = model

        if context.algorithm is not None:
            context.algorithm.model = model

        owner = context.owner
        if owner is not None:
            if owner.trainer is not None:
                owner.trainer.rate = rate
                owner.trainer.model = model
            if owner.algorithm is not None:
                owner.algorithm.model = model


class Client(simple.Client):
    """A federated learning server using the FjORD algorithm."""

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
            lifecycle_strategy=FjordLifecycleStrategy(),
            payload_strategy=payload_strategy,
            training_strategy=training_strategy,
            reporting_strategy=reporting_strategy,
            communication_strategy=communication_strategy,
        )
