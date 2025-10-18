"""
The client for system-heterogeneous federated learning through architecture search.
"""

from plato.clients import simple
from plato.clients.strategies import DefaultLifecycleStrategy
from plato.config import Config


class SysHeteroFLLifecycleStrategy(DefaultLifecycleStrategy):
    """Lifecycle strategy that adapts the client model to server-provided configs."""

    _STATE_KEY = "sysheterofl"

    @staticmethod
    def _state(context):
        return context.state.setdefault(SysHeteroFLLifecycleStrategy._STATE_KEY, {})

    def process_server_response(self, context, server_response) -> None:
        super().process_server_response(context, server_response)
        state = self._state(context)
        state["config"] = server_response.get("config")
        self._apply_if_ready(context, state)

    def configure(self, context) -> None:
        super().configure(context)
        state = self._state(context)
        self._apply_if_ready(context, state)

    def _apply_if_ready(self, context, state) -> None:
        config = state.get("config")
        if config is None:
            return

        model_factory = context.custom_model or context.model
        if model_factory is None:
            return

        model = model_factory(
            configs=config,
            **Config().parameters.client_model._asdict(),
        )

        if context.algorithm is not None:
            context.algorithm.model = model
        if context.trainer is not None:
            context.trainer.model = model

        owner = context.owner
        if owner is not None:
            if owner.algorithm is not None:
                owner.algorithm.model = model
            if owner.trainer is not None:
                owner.trainer.model = model


def create_client(
    *,
    model=None,
    datasource=None,
    algorithm=None,
    trainer=None,
    callbacks=None,
    trainer_callbacks=None,
):
    """Build a SysHeteroFL client configured with dynamic model adaptation."""
    client = simple.Client(
        model=model,
        datasource=datasource,
        algorithm=algorithm,
        trainer=trainer,
        callbacks=callbacks,
        trainer_callbacks=trainer_callbacks,
    )

    client._configure_composable(
        lifecycle_strategy=SysHeteroFLLifecycleStrategy(),
        payload_strategy=client.payload_strategy,
        training_strategy=client.training_strategy,
        reporting_strategy=client.reporting_strategy,
        communication_strategy=client.communication_strategy,
    )

    return client


# Maintain compatibility for imports expecting a Client callable/class.
Client = create_client
