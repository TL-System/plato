"""
MPC-specific client strategies layered on top of the default pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, cast

from plato.clients.strategies.defaults import (
    DefaultLifecycleStrategy,
    DefaultTrainingStrategy,
)
from plato.config import Config
from plato.mpc import RoundInfoStore


class MPCLifecycleStrategy(DefaultLifecycleStrategy):
    """Bind MPC runtime fields and the server's Shamir threshold.

    These protocol fields take precedence over explicit processor kwargs;
    other processor-specific entries are preserved.
    """

    def _build_processor_kwargs(self, context) -> dict[str, dict[str, Any]]:
        processor_kwargs = getattr(context, "processor_kwargs", {}) or {}
        if not isinstance(processor_kwargs, dict):
            processor_kwargs = {}
        round_store_obj = getattr(context, "round_store", None)
        if round_store_obj is None:
            raise RuntimeError("round_store must be available in the client context.")
        round_store = cast(RoundInfoStore, round_store_obj)

        outbound_processors = getattr(Config().clients, "outbound_processors", [])
        debug_artifacts = getattr(context, "debug_artifacts", False)

        for name in outbound_processors or []:
            if name.startswith("mpc_model_encrypt"):
                processor_kwargs.setdefault(name, {})
                processor_kwargs[name].update(
                    {
                        "client_id": context.client_id,
                        "round_store": round_store,
                        "debug_artifacts": debug_artifacts,
                    }
                )
                if name == "mpc_model_encrypt_shamir":
                    # Both ends must use the server's degree, including None's
                    # existing participant-dependent default threshold.
                    processor_kwargs[name]["threshold"] = getattr(
                        Config().server, "mpc_shamir_threshold", None
                    )
        return processor_kwargs

    def configure(self, context) -> None:
        context.processor_kwargs = self._build_processor_kwargs(context)
        super().configure(context)


class MPCTrainingStrategy(DefaultTrainingStrategy):
    """Training strategy recording client sample counts in the round store."""

    def __init__(self, round_store: RoundInfoStore) -> None:
        super().__init__()
        self.round_store = round_store

    async def train(self, context):
        round_number = self.round_store.load_state().round_number
        report, payload = await super().train(context)
        num_samples = getattr(report, "num_samples", 0)
        self.round_store.record_client_samples(
            context.client_id, num_samples, round_number=round_number
        )
        return report, payload
