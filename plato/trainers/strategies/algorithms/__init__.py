"""
Algorithm-specific strategy implementations.

This package contains concrete strategy implementations for specific
federated learning algorithms. These strategies can be composed to create
trainers that implement various FL algorithms without inheritance.

Available Strategies:
    FedProx:
        - FedProxLossStrategy: Loss with proximal term
        - FedProxLossStrategyFromConfig: Config-based variant

    SCAFFOLD:
        - SCAFFOLDUpdateStrategy: Control variate management
        - SCAFFOLDUpdateStrategyV2: Alternative implementation (Option 1)

    FedDyn:
        - FedDynLossStrategy: Dynamic regularization loss
        - FedDynUpdateStrategy: State management
        - FedDynLossStrategyFromConfig: Config-based variant

    LG-FedAvg:
        - LGFedAvgStepStrategy: Dual forward/backward passes
        - LGFedAvgStepStrategyFromConfig: Config-based variant
        - LGFedAvgStepStrategyAuto: Auto layer detection

    FedMos:
        - FedMosOptimizer: Double momentum optimizer
        - FedMosOptimizerStrategy: Optimizer strategy
        - FedMosUpdateStrategy: State management
        - FedMosOptimizerStrategyFromConfig: Config-based variant
        - FedMosStepStrategy: Training step with momentum update

    Personalized FL:
        - FedPerUpdateStrategy: FedPer personalization
        - FedPerUpdateStrategyFromConfig: Config-based variant
        - FedRepUpdateStrategy: FedRep representation learning
        - FedRepUpdateStrategyFromConfig: Config-based variant

    APFL:
        - APFLUpdateStrategy: Dual model management
        - APFLStepStrategy: Dual model training
        - APFLUpdateStrategyFromConfig: Config-based variant

    Ditto:
        - DittoUpdateStrategy: Personalized model training
        - DittoUpdateStrategyFromConfig: Config-based variant

Example:
    >>> from plato.trainers.composable import ComposableTrainer
    >>> from plato.trainers.strategies.algorithms import (
    ...     FedProxLossStrategy,
    ...     SCAFFOLDUpdateStrategy,
    ...     LGFedAvgStepStrategy
    ... )
    >>>
    >>> # Create trainer with FedProx
    >>> fedprox_trainer = ComposableTrainer(
    ...     loss_strategy=FedProxLossStrategy(mu=0.01)
    ... )
    >>>
    >>> # Create trainer with SCAFFOLD
    >>> scaffold_trainer = ComposableTrainer(
    ...     model_update_strategy=SCAFFOLDUpdateStrategy()
    ... )
    >>>
    >>> # Create trainer with LG-FedAvg
    >>> lgfedavg_trainer = ComposableTrainer(
    ...     training_step_strategy=LGFedAvgStepStrategy(
    ...         global_layer_names=['conv', 'fc1'],
    ...         local_layer_names=['fc2']
    ...     )
    ... )
"""

# FedProx strategies
# APFL strategies
from plato.trainers.strategies.algorithms.apfl_strategy import (
    APFLStepStrategy,
    APFLUpdateStrategy,
    APFLUpdateStrategyFromConfig,
)

# Ditto strategies
from plato.trainers.strategies.algorithms.ditto_strategy import (
    DittoUpdateStrategy,
    DittoUpdateStrategyFromConfig,
)

# FedDyn strategies
from plato.trainers.strategies.algorithms.feddyn_strategy import (
    FedDynLossStrategy,
    FedDynLossStrategyFromConfig,
    FedDynUpdateStrategy,
)

# FedMos strategies
from plato.trainers.strategies.algorithms.fedmos_strategy import (
    FedMosOptimizer,
    FedMosOptimizerStrategy,
    FedMosOptimizerStrategyFromConfig,
    FedMosStepStrategy,
    FedMosUpdateStrategy,
)
from plato.trainers.strategies.algorithms.fedprox_strategy import (
    FedProxLossStrategy,
    FedProxLossStrategyFromConfig,
)

# LG-FedAvg strategies
from plato.trainers.strategies.algorithms.lgfedavg_strategy import (
    LGFedAvgStepStrategy,
    LGFedAvgStepStrategyAuto,
    LGFedAvgStepStrategyFromConfig,
)

# Personalized FL strategies (FedPer, FedRep)
from plato.trainers.strategies.algorithms.personalized_fl_strategy import (
    FedPerUpdateStrategy,
    FedPerUpdateStrategyFromConfig,
    FedRepUpdateStrategy,
    FedRepUpdateStrategyFromConfig,
)

# SCAFFOLD strategies
from plato.trainers.strategies.algorithms.scaffold_strategy import (
    SCAFFOLDUpdateStrategy,
    SCAFFOLDUpdateStrategyV2,
)

__all__ = [
    # FedProx
    "FedProxLossStrategy",
    "FedProxLossStrategyFromConfig",
    # SCAFFOLD
    "SCAFFOLDUpdateStrategy",
    "SCAFFOLDUpdateStrategyV2",
    # FedDyn
    "FedDynLossStrategy",
    "FedDynLossStrategyFromConfig",
    "FedDynUpdateStrategy",
    # LG-FedAvg
    "LGFedAvgStepStrategy",
    "LGFedAvgStepStrategyFromConfig",
    "LGFedAvgStepStrategyAuto",
    # FedMos
    "FedMosOptimizer",
    "FedMosOptimizerStrategy",
    "FedMosOptimizerStrategyFromConfig",
    "FedMosStepStrategy",
    "FedMosUpdateStrategy",
    # Personalized FL
    "FedPerUpdateStrategy",
    "FedPerUpdateStrategyFromConfig",
    "FedRepUpdateStrategy",
    "FedRepUpdateStrategyFromConfig",
    # APFL
    "APFLUpdateStrategy",
    "APFLStepStrategy",
    "APFLUpdateStrategyFromConfig",
    # Ditto
    "DittoUpdateStrategy",
    "DittoUpdateStrategyFromConfig",
]
