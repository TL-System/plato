"""
Model update strategy implementations.

This module provides default and common model update strategies for
the composable trainer architecture. These strategies handle state
management and model modifications during training.
"""

from typing import Any, Dict

from plato.trainers.strategies.base import ModelUpdateStrategy, TrainingContext


class NoOpUpdateStrategy(ModelUpdateStrategy):
    """
    No-op model update strategy that does nothing.

    This is the default when no model update strategy is specified.
    It provides all the hooks but doesn't implement any custom logic.

    Example:
        >>> strategy = NoOpUpdateStrategy()
        >>> trainer = ComposableTrainer(model_update_strategy=strategy)
    """

    def on_train_start(self, context: TrainingContext) -> None:
        """No-op: called at start of training."""
        pass

    def on_train_end(self, context: TrainingContext) -> None:
        """No-op: called at end of training."""
        pass

    def before_step(self, context: TrainingContext) -> None:
        """No-op: called before each training step."""
        pass

    def after_step(self, context: TrainingContext) -> None:
        """No-op: called after each training step."""
        pass

    def get_update_payload(self, context: TrainingContext) -> dict[str, Any]:
        """No-op: return empty payload."""
        return {}


class StateTrackingUpdateStrategy(ModelUpdateStrategy):
    """
    Simple state tracking strategy that counts steps and epochs.

    This can be used as a base for more complex strategies or for
    debugging purposes.

    Example:
        >>> strategy = StateTrackingUpdateStrategy()
        >>> trainer = ComposableTrainer(model_update_strategy=strategy)
    """

    def __init__(self):
        """Initialize state tracking counters."""
        self.total_steps = 0
        self.epoch_steps = 0
        self.round_number = 0

    def setup(self, context: TrainingContext) -> None:
        """Initialize counters."""
        self.total_steps = 0
        self.epoch_steps = 0
        self.round_number = 0

    def on_train_start(self, context: TrainingContext) -> None:
        """Reset epoch counter and increment round."""
        self.epoch_steps = 0
        self.round_number += 1

    def after_step(self, context: TrainingContext) -> None:
        """Increment step counters."""
        self.total_steps += 1
        self.epoch_steps += 1

    def get_update_payload(self, context: TrainingContext) -> dict[str, Any]:
        """Return step statistics."""
        return {
            "total_steps": self.total_steps,
            "epoch_steps": self.epoch_steps,
            "round_number": self.round_number,
        }


class CompositeUpdateStrategy(ModelUpdateStrategy):
    """
    Composite strategy that combines multiple update strategies.

    This allows you to use multiple update strategies together,
    useful when different algorithms need to be combined.

    Args:
        strategies: List of model update strategies to combine

    Example:
        >>> strategy1 = StateTrackingUpdateStrategy()
        >>> strategy2 = CustomUpdateStrategy()
        >>> composite = CompositeUpdateStrategy([strategy1, strategy2])
        >>> trainer = ComposableTrainer(model_update_strategy=composite)
    """

    def __init__(self, strategies: list):
        """Initialize with list of strategies."""
        self.strategies = strategies

    def setup(self, context: TrainingContext) -> None:
        """Setup all strategies."""
        for strategy in self.strategies:
            strategy.setup(context)

    def on_train_start(self, context: TrainingContext) -> None:
        """Call on_train_start for all strategies."""
        for strategy in self.strategies:
            strategy.on_train_start(context)

    def on_client_id_changed(self, context: TrainingContext) -> None:
        """Propagate ownership changes to each composed strategy."""
        for strategy in self.strategies:
            strategy.on_client_id_changed(context)

    def on_train_end(self, context: TrainingContext) -> None:
        """Call on_train_end for all strategies."""
        for strategy in self.strategies:
            strategy.on_train_end(context)

    def on_train_cleanup(self, context: TrainingContext, successful: bool) -> None:
        """Release each component's run resources."""
        for strategy in self.strategies:
            strategy.on_train_cleanup(context, successful)

    def on_train_result_accepted(self, context: TrainingContext) -> None:
        """Commit each component's staged successful result."""
        for strategy in self.strategies:
            strategy.on_train_result_accepted(context)

    @property
    def requires_worker_state(self) -> bool:
        return any(strategy.requires_worker_state for strategy in self.strategies)

    def get_worker_state(self, context: TrainingContext) -> list:
        return [strategy.get_worker_state(context) for strategy in self.strategies]

    def load_worker_state(self, state: Any, context: TrainingContext) -> None:
        if not isinstance(state, list) or len(state) != len(self.strategies):
            raise ValueError("Training worker returned mismatched strategy state.")
        for strategy, component_state in zip(self.strategies, state):
            strategy.load_worker_state(component_state, context)

    def before_step(self, context: TrainingContext) -> None:
        """Call before_step for all strategies."""
        for strategy in self.strategies:
            strategy.before_step(context)

    def after_step(self, context: TrainingContext) -> None:
        """Call after_step for all strategies."""
        for strategy in self.strategies:
            strategy.after_step(context)

    def get_update_payload(self, context: TrainingContext) -> dict[str, Any]:
        """Merge payloads from all strategies."""
        merged_payload = {}
        for strategy in self.strategies:
            payload = strategy.get_update_payload(context)
            merged_payload.update(payload)
        return merged_payload

    def teardown(self, context: TrainingContext) -> None:
        """Teardown all strategies."""
        for strategy in self.strategies:
            strategy.teardown(context)
