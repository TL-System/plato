"""
Client implementation for the MOON aggregation example.

The client composes the new trainer and provides frozen copies of the global
and historical local models required by the MOON contrastive loss.
"""

from __future__ import annotations

import copy
from typing import List, Optional

from moon_model import Model as MoonModel
from moon_trainer import Trainer as MoonTrainer

from plato.clients import simple
from plato.clients.strategies.defaults import DefaultTrainingStrategy
from plato.config import Config


class MoonTrainingStrategy(DefaultTrainingStrategy):
    """Attach frozen global and historical models to the trainer context."""

    def __init__(self, buffer_size: int = 1):
        super().__init__()
        self.buffer_size = buffer_size
        self.history: List = []

    async def train(self, context):
        trainer = context.trainer
        device = trainer.device

        # Clone the current global model before local updates begin.
        global_clone = trainer.clone_model()
        global_clone.to(device)
        context.trainer.context.state["moon_global_model"] = global_clone

        # Prepare historical client models as negatives.
        historical_models = []
        for cached in self.history:
            model_copy = copy.deepcopy(cached)
            model_copy.to(device)
            historical_models.append(model_copy)

        context.trainer.context.state["moon_prev_models"] = historical_models

        report, weights = await super().train(context)

        # Move temporary models back to CPU before releasing references.
        global_clone.to("cpu")
        for model_copy in historical_models:
            model_copy.to("cpu")

        # Update history with the freshly trained local model.
        trained_snapshot = trainer.clone_model()
        self.history.append(trained_snapshot)
        if len(self.history) > self.buffer_size:
            self.history.pop(0)

        # Clean references to free device memory.
        context.trainer.context.state.pop("moon_global_model", None)
        context.trainer.context.state.pop("moon_prev_models", None)

        return report, weights


class Client(simple.Client):
    """MOON client wiring the customised trainer and training strategy."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        trainer_callbacks=None,
        history_size: Optional[int] = None,
    ):
        model = model or MoonModel
        trainer = trainer or MoonTrainer

        if history_size is None:
            algorithm_cfg = getattr(Config(), "algorithm", None)
            history_size = (
                getattr(algorithm_cfg, "history_size", 1) if algorithm_cfg else 1
            )

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            trainer_callbacks=trainer_callbacks,
        )

        self._configure_composable(
            lifecycle_strategy=self.lifecycle_strategy,
            payload_strategy=self.payload_strategy,
            training_strategy=MoonTrainingStrategy(buffer_size=history_size),
            reporting_strategy=self.reporting_strategy,
            communication_strategy=self.communication_strategy,
        )
