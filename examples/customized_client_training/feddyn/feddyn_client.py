"""Dedicated FedDyn adapters for the actual composable payload path."""

import copy

from plato.clients import simple
from plato.clients.strategies.defaults import (
    DefaultLifecycleStrategy,
    DefaultTrainingStrategy,
)
from plato.trainers.strategies.algorithms.feddyn_strategy import (
    settings_from_config,
    validate_dispatch,
)


class FedDynLifecycle(DefaultLifecycleStrategy):
    """Bind payload identity to the active server assignment header."""

    def process_server_response(self, context, server_response):
        header = server_response.get("feddyn")
        if not isinstance(header, dict):
            raise ValueError("FedDyn requires its dedicated server assignment.")
        context.state["feddyn_assignment"] = copy.deepcopy(header)


class FedDynTraining(DefaultTrainingStrategy):
    """Validate full downlinks and refuse failed attempts on the live path."""

    def load_payload(self, context, server_payload):
        trainer = context.trainer
        state = validate_dispatch(
            trainer.model,
            server_payload,
            settings_from_config(),
            context.client_id,
            context.current_round,
        )
        assignment = context.state.get("feddyn_assignment")
        expected = {
            k: state["metadata"][k]
            for k in ("version", "run_id", "round", "client_id", "dispatch_token")
        }
        if assignment != expected:
            raise ValueError(
                "FedDyn payload differs from the active assignment header."
            )
        # Full validation precedes either model or strategy mutation.
        before = copy.deepcopy(trainer.model.state_dict())
        try:
            context.algorithm.load_weights(state["baseline"])
        except BaseException:
            trainer.model.load_state_dict(before)
            raise
        trainer.model_update_strategy.install_dispatch(state, trainer.context)

    async def train(self, context):
        context.state.pop("training_error", None)
        try:
            report, weights = await super().train(context)
            error = context.state.pop("training_error", None)
            if error is not None:
                raise error
            metadata = context.trainer.model_update_strategy.get_update_payload(
                context.trainer.context
            )
            report.num_samples = metadata["num_samples"]
            return report, [weights, metadata]
        except BaseException:
            context.trainer.model_update_strategy.on_train_cleanup(
                context.trainer.context, successful=False
            )
            raise


def create_client(
    *, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
):
    """Build the dedicated client while retaining default transport/reporting."""
    if trainer is None:
        from feddyn_trainer import Trainer

        trainer = Trainer
    client = simple.Client(
        model=model,
        datasource=datasource,
        algorithm=algorithm,
        trainer=trainer,
        callbacks=callbacks,
    )
    client._configure_composable(
        lifecycle_strategy=FedDynLifecycle(),
        payload_strategy=client.payload_strategy,
        training_strategy=FedDynTraining(),
        reporting_strategy=client.reporting_strategy,
        communication_strategy=client.communication_strategy,
    )
    return client


Client = create_client
