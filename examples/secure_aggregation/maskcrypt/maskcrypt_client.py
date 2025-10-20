"""
A MaskCrypt client with selective homomorphic encryption support.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Tuple

import maskcrypt_utils
from maskcrypt_algorithm import Algorithm as MaskCryptAlgorithm

from plato.clients import simple
from plato.clients.strategies import DefaultTrainingStrategy
from plato.clients.strategies.base import ClientContext
from plato.config import Config


class MaskCryptTrainingStrategy(DefaultTrainingStrategy):
    """Training strategy implementing the MaskCrypt alternating workflow."""

    _STATE_KEY = "maskcrypt"

    def __init__(
        self,
        *,
        encrypt_ratio: float,
        random_mask: bool,
        checkpoint_path: str,
        attack_prep_dir: str,
    ) -> None:
        super().__init__()
        self.encrypt_ratio = encrypt_ratio
        self.random_mask = random_mask
        self.checkpoint_path = checkpoint_path
        self.attack_prep_dir = attack_prep_dir

    def setup(self, context: ClientContext) -> None:
        """Initialise persistent state for the MaskCrypt workflow."""
        super().setup(context)
        state = self._state(context)
        state.setdefault("model_buffer", {})
        state.setdefault("final_mask", None)
        state.setdefault("checkpoint_path", self.checkpoint_path)
        state.setdefault("attack_prep_dir", self.attack_prep_dir)
        state.setdefault("encrypt_ratio", self.encrypt_ratio)
        state.setdefault("random_mask", self.random_mask)

    def load_payload(self, context: ClientContext, server_payload: Any) -> None:
        """Store inbound payload or delegate weight loading based on the round."""
        state = self._state(context)

        if context.current_round % 2 != 0:
            state["final_mask"] = None
            context.owner.final_mask = None
            super().load_payload(context, server_payload)
            return

        state["final_mask"] = server_payload
        context.owner.final_mask = server_payload

    async def train(self, context: ClientContext) -> Tuple[Any, Any]:
        """Alternate between mask proposal computation and weight submission."""
        if context.current_round % 2 != 0:
            report, weights = await super().train(context)
            gradients = context.trainer.get_gradient()
            mask = self._compute_mask(context, weights, gradients)
            self._state(context)["model_buffer"][context.client_id] = (
                report,
                weights,
            )
            return report, mask

        cached = self._state(context)["model_buffer"].pop(context.client_id, None)
        if cached is None:
            raise RuntimeError("No cached weights available for MaskCrypt client.")

        report, weights = cached
        # Ensure ordering in asynchronous queues remains stable.
        report.training_time = max(getattr(report, "training_time", 0.0), 0.001)
        report.comm_time = time.time()

        return report, weights

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _state(self, context: ClientContext) -> Dict[str, Any]:
        return context.state.setdefault(self._STATE_KEY, {})

    def _get_estimate_path(self, client_id: int) -> str:
        model_name = Config().trainer.model_name
        return os.path.join(
            self.checkpoint_path,
            self.attack_prep_dir,
            f"{model_name}_est_{client_id}.pth",
        )

    def _get_plain_path(self, client_id: int) -> str:
        model_name = Config().trainer.model_name
        return os.path.join(
            self.checkpoint_path,
            self.attack_prep_dir,
            f"{model_name}_plain_{client_id}.pth",
        )

    def _compute_mask(
        self,
        context: ClientContext,
        latest_weights,
        gradients,
    ):
        """Compute the selective encryption mask for the current client."""
        client_id = context.client_id

        algorithm = context.algorithm
        latest_flat = algorithm.flatten_weights(latest_weights)
        gradients_flat = algorithm.flatten_gradients(gradients)

        plain_path = self._get_plain_path(client_id)
        algorithm.store_plain_weights(plain_path, latest_flat)

        estimate = maskcrypt_utils.get_est(self._get_estimate_path(client_id))
        exposed_flat = algorithm.prepare_exposed_weights(estimate, latest_flat)

        return algorithm.compute_mask(
            latest_flat=latest_flat,
            gradients_flat=gradients_flat,
            exposed_flat=exposed_flat,
            encrypt_ratio=self.encrypt_ratio,
            random_mask=self.random_mask,
        )


class MaskCryptClientProxy(simple.Client):
    """Client variant exposing MaskCrypt state via a convenient property."""

    @property
    def final_mask(self):
        return self._context.state.get("maskcrypt", {}).get("final_mask")

    @final_mask.setter
    def final_mask(self, value):
        self._context.state.setdefault("maskcrypt", {})["final_mask"] = value


def create_client(
    *,
    model=None,
    datasource=None,
    algorithm=None,
    trainer=None,
    callbacks=None,
):
    """Build a MaskCrypt client configured with selective encryption."""
    client = MaskCryptClientProxy(
        model=model,
        datasource=datasource,
        algorithm=algorithm or MaskCryptAlgorithm,
        trainer=trainer,
        callbacks=callbacks,
    )

    client.encrypt_ratio = Config().clients.encrypt_ratio
    client.random_mask = Config().clients.random_mask
    client.attack_prep_dir = (
        f"{Config().data.datasource}_{Config().trainer.model_name}"
        f"_{client.encrypt_ratio}"
    )
    if client.random_mask:
        client.attack_prep_dir += "_random"

    client.checkpoint_path = Config().params["checkpoint_path"]

    state = client._context.state.setdefault("maskcrypt", {})
    state.setdefault("model_buffer", {})
    state.setdefault("final_mask", None)

    client.final_mask = None

    client._configure_composable(
        lifecycle_strategy=client.lifecycle_strategy,
        payload_strategy=client.payload_strategy,
        training_strategy=MaskCryptTrainingStrategy(
            encrypt_ratio=client.encrypt_ratio,
            random_mask=client.random_mask,
            checkpoint_path=client.checkpoint_path,
            attack_prep_dir=client.attack_prep_dir,
        ),
        reporting_strategy=client.reporting_strategy,
        communication_strategy=client.communication_strategy,
    )

    return client


# Maintain compatibility for imports expecting a Client callable/class.
Client = create_client
