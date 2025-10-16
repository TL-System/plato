"""
Client implementation for MPC-enabled training sessions.
"""

from __future__ import annotations

from typing import Optional

from plato.clients import simple
from plato.clients.strategies.defaults import (
    DefaultCommunicationStrategy,
    DefaultPayloadStrategy,
    DefaultReportingStrategy,
)
from plato.clients.strategies.mpc import MPCLifecycleStrategy, MPCTrainingStrategy
from plato.mpc import RoundInfoStore


class Client(simple.Client):
    """Extends the simple client with MPC-aware strategies."""

    def __init__(
        self,
        *,
        round_store_lock: Optional[object] = None,
        debug_artifacts: bool = False,
        **kwargs,
    ):
        self._round_store_lock = round_store_lock
        self._debug_artifacts = debug_artifacts
        super().__init__(**kwargs)

        self.round_store = RoundInfoStore.from_config(lock=self._round_store_lock)
        self._context.round_store = self.round_store
        self._context.debug_artifacts = self._debug_artifacts

        self._configure_composable(
            lifecycle_strategy=MPCLifecycleStrategy(),
            payload_strategy=DefaultPayloadStrategy(),
            training_strategy=MPCTrainingStrategy(self.round_store),
            reporting_strategy=DefaultReportingStrategy(),
            communication_strategy=DefaultCommunicationStrategy(),
        )

    def configure(self) -> None:
        self._context.round_store = self.round_store
        self._context.processor_kwargs = getattr(self._context, "processor_kwargs", {})
        super().configure()
