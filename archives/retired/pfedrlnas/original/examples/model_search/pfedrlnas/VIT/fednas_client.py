"""
Customized Client for PerFedRLNAS.
"""

from plato.clients import simple
from plato.clients.strategies import DefaultLifecycleStrategy


class PerFedRLNASVitLifecycleStrategy(DefaultLifecycleStrategy):
    """Lifecycle strategy that applies server-provided subnet configurations."""

    _STATE_KEY = "pfedrlnas_vit"

    @staticmethod
    def _state(context):
        return context.state.setdefault(PerFedRLNASVitLifecycleStrategy._STATE_KEY, {})

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


def create_client(
    *,
    model=None,
    datasource=None,
    algorithm=None,
    trainer=None,
    callbacks=None,
    trainer_callbacks=None,
):
    """Build a PerFedRLNAS ViT client with the custom lifecycle strategy."""
    client = simple.Client(
        model=model,
        datasource=datasource,
        algorithm=algorithm,
        trainer=trainer,
        callbacks=callbacks,
        trainer_callbacks=trainer_callbacks,
    )

    client._configure_composable(
        lifecycle_strategy=PerFedRLNASVitLifecycleStrategy(),
        payload_strategy=client.payload_strategy,
        training_strategy=client.training_strategy,
        reporting_strategy=client.reporting_strategy,
        communication_strategy=client.communication_strategy,
    )

    return client


# Maintain compatibility for imports expecting a Client callable/class.
Client = create_client
