"""
Algorithm-specific strategy implementations.

This package will contain concrete strategy implementations for specific
federated learning algorithms like FedProx, SCAFFOLD, FedDyn, etc.

As algorithms are migrated from inheritance-based to composition-based design,
their strategy implementations will be added to this package.

Example structure (to be implemented in Phase 3):
    - fedprox_strategy.py: FedProx loss strategy
    - scaffold_strategy.py: SCAFFOLD control variate strategy
    - feddyn_strategy.py: FedDyn loss and update strategies
    - lgfedavg_strategy.py: LG-FedAvg training step strategy
    - and more...

Usage:
    >>> from plato.trainers.strategies.implementations import FedProxLossStrategy
    >>> from plato.trainers.composable import ComposableTrainer
    >>>
    >>> trainer = ComposableTrainer(
    ...     loss_strategy=FedProxLossStrategy(mu=0.01)
    ... )
"""

# This package is currently empty and will be populated in Phase 3
# of the refactoring roadmap with algorithm-specific implementations.

__all__ = []
