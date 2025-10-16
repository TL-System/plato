"""
MPC-specific client strategies layered on top of the default pipeline.
"""

from __future__ import annotations

from typing import Dict

from plato.clients.strategies.defaults import (
    DefaultLifecycleStrategy,
    DefaultTrainingStrategy,
)
from plato.config import Config
from plato.mpc import RoundInfoStore


class MPCLifecycleStrategy(DefaultLifecycleStrategy):
    """Lifecycle strategy wiring the MPC round store into processor kwargs."""

    def _build_processor_kwargs(self, context) -> Dict[str, Dict]:
        processor_kwargs = getattr(context, "processor_kwargs", {}) or {}
        round_store: RoundInfoStore = getattr(context, "round_store", None)
        if round_store is None:
            raise RuntimeError("round_store must be available in the client context.")

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
        report, payload = await super().train(context)
        num_samples = getattr(report, "num_samples", 0)
        if num_samples:
            self.round_store.record_client_samples(context.client_id, num_samples)
        return report, payload
