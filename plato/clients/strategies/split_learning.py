"""
Split learning client strategies.

These strategies reproduce the specialised split-learning workflow using the
composable client runtime. Clients alternate between feature extraction and
gradient application stages while maintaining per-client contexts.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from typing import Any, Dict, Tuple

from plato.clients.strategies.base import ClientContext, TrainingStrategy
from plato.config import Config
from plato.utils import fonts


class SplitLearningTrainingStrategy(TrainingStrategy):
    """Training strategy that orchestrates the split learning lifecycle."""

    _STATE_KEY = "split_learning"

    def setup(self, context: ClientContext) -> None:
        """Initialise split learning state."""
        state = self._state(context)
        state.setdefault("contexts", {})
        state.setdefault("original_weights", None)
        state.setdefault("static_sampler", None)
        iterations = getattr(Config().clients, "iteration", 1)
        state.setdefault("iterations", iterations)
        state.setdefault("iter_left", iterations)
        state.setdefault("incoming_payload", None)

    def load_payload(self, context: ClientContext, server_payload: Any) -> None:
        """Store inbound payload for later processing."""
        self._state(context)["incoming_payload"] = server_payload

    async def train(self, context: ClientContext) -> Tuple[Any, Any]:
        """Handle split learning training steps."""
        state = self._state(context)
        inbound = state.pop("incoming_payload", None)

        if inbound is None:
            raise RuntimeError("No inbound payload available for split learning.")

        payload, info = inbound

        if info == "prompt":
            return self._handle_prompt(context, state)
        if info == "gradients":
            return self._handle_gradients(context, state, payload)

        raise ValueError(f"Unsupported split learning payload type: {info}")

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _state(self, context: ClientContext) -> Dict[str, Any]:
        """Return the mutable state dictionary for split learning."""
        return context.state.setdefault(self._STATE_KEY, {})

    def _load_context(self, context: ClientContext, state: Dict[str, Any]) -> None:
        """Restore model weights and samplers for the current client."""
        client_id = context.client_id
        contexts = state["contexts"]

        if client_id in contexts:
            weights, sampler = contexts.pop(client_id)
            context.algorithm.load_weights(weights)
            state["static_sampler"] = sampler
            return

        if state["original_weights"] is None:
            state["original_weights"] = context.algorithm.extract_weights()

        context.algorithm.load_weights(state["original_weights"])

        sampler = getattr(context, "sampler", None)
        if sampler is not None and hasattr(sampler, "get"):
            state["static_sampler"] = sampler.get()
        else:
            state["static_sampler"] = sampler

    def _save_context(self, context: ClientContext, state: Dict[str, Any]) -> None:
        """Persist model weights and sampler for the current client."""
        sampler = state["static_sampler"]
        contexts = state["contexts"]
        contexts[context.client_id] = (
            context.algorithm.extract_weights(),
            sampler,
        )

    def _extract_features(
        self, context: ClientContext, state: Dict[str, Any]
    ) -> Tuple[Any, Any]:
        """Run feature extraction until the cut layer."""
        iterations = state["iterations"]
        iter_left = state["iter_left"]
        round_number = iterations - iter_left + 1

        logging.warning(
            fonts.colourize(
                f"[{context.owner}] Started split learning in round "
                f"#{round_number}/{iterations} (Global round {context.current_round})."
            )
        )

        features, training_time = context.algorithm.extract_features(
            context.trainset, state["static_sampler"]
        )

        sampler = context.sampler
        num_samples = sampler.num_samples() if sampler is not None else 0

        report = SimpleNamespace(
            client_id=context.client_id,
            num_samples=num_samples,
            accuracy=0,
            training_time=training_time,
            comm_time=time.time(),
            update_response=False,
            type="features",
        )

        return report, features

    def _handle_prompt(
        self, context: ClientContext, state: Dict[str, Any]
    ) -> Tuple[Any, Any]:
        """Respond to a prompt by performing feature extraction."""
        self._load_context(context, state)
        report, features = self._extract_features(context, state)
        return report, features

    def _handle_gradients(
        self,
        context: ClientContext,
        state: Dict[str, Any],
        gradients: Any,
    ) -> Tuple[Any, Any]:
        """Apply gradients and optionally continue feature extraction."""
        training_time = context.algorithm.complete_train(gradients)
        weights = context.algorithm.extract_weights()

        state["iter_left"] -= 1
        iterations = state["iterations"]

        sampler = context.sampler
        num_samples = sampler.num_samples() if sampler is not None else 0

        if state["iter_left"] == 0:
            report = SimpleNamespace(
                client_id=context.client_id,
                num_samples=num_samples,
                accuracy=0,
                training_time=training_time,
                comm_time=time.time(),
                update_response=False,
                type="weights",
            )
            outbound = weights
            state["iter_left"] = iterations
        else:
            report, outbound = self._extract_features(context, state)
            report.training_time += training_time

        self._save_context(context, state)
        return report, outbound
