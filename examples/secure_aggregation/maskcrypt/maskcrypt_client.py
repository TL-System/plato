"""
A MaskCrypt client with selective homomorphic encryption support.
"""

from __future__ import annotations

import os
import pickle
import random
import time
from typing import Any, Dict, Tuple

import maskcrypt_utils
import torch

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

    def _ensure_attack_dir(self) -> None:
        target_dir = os.path.join(self.checkpoint_path, self.attack_prep_dir)
        os.makedirs(target_dir, exist_ok=True)

    def _get_exposed_weights(self, client_id: int) -> torch.Tensor:
        """Load the exposed weights tracked so far."""
        estimate = maskcrypt_utils.get_est(self._get_estimate_path(client_id))
        if estimate is None:
            return torch.tensor([])
        return torch.tensor(estimate)

    def _store_plain_weights(self, client_id: int, latest_flat: torch.Tensor) -> None:
        """Persist the plain (unencrypted) weights for analysis."""
        self._ensure_attack_dir()
        model_name = Config().trainer.model_name
        plain_path = os.path.join(
            self.checkpoint_path,
            self.attack_prep_dir,
            f"{model_name}_plain_{client_id}.pth",
        )
        with open(plain_path, "wb") as plain_file:
            pickle.dump(latest_flat.cpu(), plain_file)

    def _compute_mask(
        self,
        context: ClientContext,
        latest_weights: Dict[str, torch.Tensor],
        gradients: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the selective encryption mask for the current client."""
        client_id = context.client_id

        exposed_flat = self._get_exposed_weights(client_id)
        latest_flat = torch.cat(
            [torch.flatten(latest_weights[name]).cpu() for name in latest_weights]
        )
        self._store_plain_weights(client_id, latest_flat)

        if self.random_mask:
            mask_len = int(self.encrypt_ratio * len(latest_flat))
            if mask_len <= 0:
                return torch.tensor([], dtype=torch.long)
            selected = random.sample(range(len(latest_flat)), mask_len)
            return torch.tensor(selected, dtype=torch.long)

        if exposed_flat.numel() == 0:
            exposed_flat = torch.zeros_like(latest_flat)
        else:
            exposed_flat = exposed_flat.cpu()

        grad_flat = torch.cat(
            [torch.flatten(gradients[name]).cpu() for name in gradients]
        )
        grad_flat = grad_flat.to(latest_flat.dtype)

        delta = exposed_flat.to(latest_flat.dtype) - latest_flat
        product = delta * grad_flat

        _, indices = torch.sort(product, descending=True)
        mask_len = int(self.encrypt_ratio * len(indices))
        if mask_len <= 0:
            return torch.tensor([], dtype=torch.long)

        topk = indices[:mask_len].clone().detach()
        return topk.to(dtype=torch.long)


class Client(simple.Client):
    """A MaskCrypt client with bespoke training strategy."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        self.encrypt_ratio = Config().clients.encrypt_ratio
        self.random_mask = Config().clients.random_mask
        self.attack_prep_dir = (
            f"{Config().data.datasource}_{Config().trainer.model_name}"
            f"_{self.encrypt_ratio}"
        )
        if self.random_mask:
            self.attack_prep_dir += "_random"

        self.checkpoint_path = Config().params["checkpoint_path"]

        state = self._context.state.setdefault("maskcrypt", {})
        state.setdefault("model_buffer", {})
        state.setdefault("final_mask", None)

        self.final_mask = None

        self._configure_composable(
            lifecycle_strategy=self.lifecycle_strategy,
            payload_strategy=self.payload_strategy,
            training_strategy=MaskCryptTrainingStrategy(
                encrypt_ratio=self.encrypt_ratio,
                random_mask=self.random_mask,
                checkpoint_path=self.checkpoint_path,
                attack_prep_dir=self.attack_prep_dir,
            ),
            reporting_strategy=self.reporting_strategy,
            communication_strategy=self.communication_strategy,
        )

    @property
    def final_mask(self):
        """Expose the final mask agreed by the server for callbacks/processors."""
        return self._context.state.get("maskcrypt", {}).get("final_mask")

    @final_mask.setter
    def final_mask(self, value):
        self._context.state.setdefault("maskcrypt", {})["final_mask"] = value
