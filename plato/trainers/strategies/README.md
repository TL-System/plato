# Trainer Strategies

**Composable Trainer Architecture for Plato Federated Learning Framework**

This package provides strategy interfaces and implementations for building flexible, composable trainers using the Strategy pattern and Dependency Injection instead of inheritance.

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Strategy Types](#strategy-types)
- [Usage Examples](#usage-examples)
- [Creating Custom Strategies](#creating-custom-strategies)
- [API Reference](#api-reference)

---

## Overview

### Why Strategies?

Traditional inheritance-based trainer extension has limitations:
- ❌ Tight coupling between subclasses and base class
- ❌ Cannot combine multiple behaviors (e.g., FedProx + SCAFFOLD)
- ❌ Difficult to test individual components
- ❌ Fragile base class problem

Strategies solve these problems:
- ✅ Composition over inheritance
- ✅ Easy to combine multiple strategies
- ✅ Each strategy is independently testable
- ✅ Clear separation of concerns

### Architecture

```
ComposableTrainer (Phase 2)
    ├── LossCriterionStrategy      # How to compute loss
    ├── OptimizerStrategy          # How to create optimizer
    ├── TrainingStepStrategy       # How to perform training step
    ├── LRSchedulerStrategy        # How to schedule learning rate
    ├── ModelUpdateStrategy        # How to manage state/updates
    └── DataLoaderStrategy         # How to load data
```

---

## Quick Start

### Installation

No installation needed - this is part of the Plato framework.

### Basic Usage (Phase 2+)

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies import (
    CrossEntropyLossStrategy,
    AdamOptimizerStrategy,
    CosineAnnealingLRSchedulerStrategy,
)

# Create trainer with custom strategies
trainer = ComposableTrainer(
    loss_strategy=CrossEntropyLossStrategy(label_smoothing=0.1),
    optimizer_strategy=AdamOptimizerStrategy(lr=0.001),
    lr_scheduler_strategy=CosineAnnealingLRSchedulerStrategy(T_max=50),
)

# Use in federated learning
from plato.clients import simple
from plato.servers import fedavg

client = simple.Client(trainer=trainer)
server = fedavg.Server(trainer=trainer)
server.run(client)
```

---

## Strategy Types

### 1. LossCriterionStrategy

**Purpose**: Customize how loss is computed

**Interface**:
```python
class LossCriterionStrategy(Strategy):
    def compute_loss(self, outputs, labels, context) -> torch.Tensor:
        """Compute loss from outputs and labels."""
        pass
```

**Implementations**:
- `CrossEntropyLossStrategy` - Classification loss
- `MSELossStrategy` - Regression loss
- `BCEWithLogitsLossStrategy` - Binary classification
- `NLLLossStrategy` - Negative log likelihood
- `CompositeLossStrategy` - Combine multiple losses
- `L2RegularizationStrategy` - Weight regularization

**Example**:
```python
loss_strategy = CrossEntropyLossStrategy(label_smoothing=0.1)
```

---

### 2. OptimizerStrategy

**Purpose**: Customize optimizer creation and configuration

**Interface**:
```python
class OptimizerStrategy(Strategy):
    def create_optimizer(self, model, context) -> torch.optim.Optimizer:
        """Create optimizer for the model."""
        pass
```

**Implementations**:
- `SGDOptimizerStrategy` - Stochastic gradient descent
- `AdamOptimizerStrategy` - Adam optimizer
- `AdamWOptimizerStrategy` - AdamW with decoupled weight decay
- `RMSpropOptimizerStrategy` - RMSprop optimizer
- `ParameterGroupOptimizerStrategy` - Different LRs per layer
- `GradientClippingOptimizerStrategy` - Gradient clipping wrapper

**Example**:
```python
optimizer_strategy = AdamOptimizerStrategy(
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.01
)
```

---

### 3. TrainingStepStrategy

**Purpose**: Customize the training step (forward + backward + optimize)

**Interface**:
```python
class TrainingStepStrategy(Strategy):
    def training_step(self, model, optimizer, examples, labels, 
                     loss_criterion, context) -> torch.Tensor:
        """Perform one training step."""
        pass
```

**Implementations**:
- `DefaultTrainingStepStrategy` - Standard training
- `GradientAccumulationStepStrategy` - Accumulate gradients
- `MixedPrecisionStepStrategy` - Automatic mixed precision (AMP)
- `GradientClippingStepStrategy` - Clip gradients
- `CustomBackwardStepStrategy` - Custom backward hook
- `MultipleForwardPassStepStrategy` - Multiple passes per batch

**Example**:
```python
training_step_strategy = MixedPrecisionStepStrategy(enabled=True)
```

---

### 4. LRSchedulerStrategy

**Purpose**: Customize learning rate scheduling

**Interface**:
```python
class LRSchedulerStrategy(Strategy):
    def create_scheduler(self, optimizer, context) -> Optional[lr_scheduler]:
        """Create LR scheduler."""
        pass
    
    def step(self, scheduler, context) -> None:
        """Step the scheduler."""
        pass
```

**Implementations**:
- `StepLRSchedulerStrategy` - Step decay
- `CosineAnnealingLRSchedulerStrategy` - Cosine annealing
- `MultiStepLRSchedulerStrategy` - Decay at milestones
- `ExponentialLRSchedulerStrategy` - Exponential decay
- `WarmupSchedulerStrategy` - Warmup + base scheduler
- And 6 more...

**Example**:
```python
lr_scheduler_strategy = CosineAnnealingLRSchedulerStrategy(T_max=50)
```

---

### 5. ModelUpdateStrategy

**Purpose**: Manage state and model updates (e.g., SCAFFOLD control variates)

**Interface**:
```python
class ModelUpdateStrategy(Strategy):
    def on_train_start(self, context) -> None: pass
    def on_train_end(self, context) -> None: pass
    def before_step(self, context) -> None: pass
    def after_step(self, context) -> None: pass
    def get_update_payload(self, context) -> Dict[str, Any]: pass
```

**Implementations**:
- `NoOpUpdateStrategy` - No-op (default)
- `StateTrackingUpdateStrategy` - Track steps/epochs
- `CompositeUpdateStrategy` - Combine multiple strategies
- Algorithm-specific (Phase 3): SCAFFOLD, FedDyn, etc.

**Example**:
```python
# Phase 3: SCAFFOLD control variates
model_update_strategy = SCAFFOLDUpdateStrategy()
```

---

### 6. DataLoaderStrategy

**Purpose**: Customize data loading

**Interface**:
```python
class DataLoaderStrategy(Strategy):
    def create_train_loader(self, trainset, sampler, batch_size, 
                           context) -> DataLoader:
        """Create training data loader."""
        pass
```

**Implementations**:
- `DefaultDataLoaderStrategy` - Standard PyTorch DataLoader
- `PrefetchDataLoaderStrategy` - Prefetch for speed
- `CustomCollateFnDataLoaderStrategy` - Custom collate function
- `DynamicBatchSizeDataLoaderStrategy` - Adjust batch size
- `ShuffleDataLoaderStrategy` - Always shuffle

**Example**:
```python
data_loader_strategy = PrefetchDataLoaderStrategy(
    prefetch_factor=4,
    num_workers=4
)
```

---

## Usage Examples

### Example 1: Simple Customization

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies import CrossEntropyLossStrategy

# Just customize loss, use defaults for everything else
trainer = ComposableTrainer(
    loss_strategy=CrossEntropyLossStrategy(label_smoothing=0.1)
)
```

### Example 2: Multiple Customizations

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies import (
    CrossEntropyLossStrategy,
    AdamWOptimizerStrategy,
    CosineAnnealingLRSchedulerStrategy,
    MixedPrecisionStepStrategy,
)

trainer = ComposableTrainer(
    loss_strategy=CrossEntropyLossStrategy(),
    optimizer_strategy=AdamWOptimizerStrategy(lr=0.001, weight_decay=0.01),
    training_step_strategy=MixedPrecisionStepStrategy(),
    lr_scheduler_strategy=CosineAnnealingLRSchedulerStrategy(T_max=50),
)
```

### Example 3: Composing Multiple Losses

```python
from plato.trainers.strategies import (
    CompositeLossStrategy,
    CrossEntropyLossStrategy,
    L2RegularizationStrategy,
)

# Combine classification loss with L2 regularization
composite_loss = CompositeLossStrategy([
    (CrossEntropyLossStrategy(), 1.0),           # weight = 1.0
    (L2RegularizationStrategy(weight=0.01), 1.0) # weight = 1.0
])

trainer = ComposableTrainer(loss_strategy=composite_loss)
```

### Example 4: Gradient Accumulation

```python
from plato.trainers.strategies import GradientAccumulationStepStrategy

# Effectively 4x batch size through gradient accumulation
training_step_strategy = GradientAccumulationStepStrategy(
    accumulation_steps=4
)

trainer = ComposableTrainer(training_step_strategy=training_step_strategy)
```

---

## Creating Custom Strategies

### Step 1: Choose Strategy Type

Decide which aspect of training you want to customize:
- Loss computation? → `LossCriterionStrategy`
- Optimizer? → `OptimizerStrategy`
- Training step? → `TrainingStepStrategy`
- LR scheduling? → `LRSchedulerStrategy`
- State management? → `ModelUpdateStrategy`
- Data loading? → `DataLoaderStrategy`

### Step 2: Implement Interface

```python
from plato.trainers.strategies.base import LossCriterionStrategy, TrainingContext
import torch
import torch.nn as nn

class MyCustomLossStrategy(LossCriterionStrategy):
    """
    My custom loss strategy.
    
    Args:
        alpha: Weight for custom term
    
    Example:
        >>> strategy = MyCustomLossStrategy(alpha=0.5)
        >>> trainer = ComposableTrainer(loss_strategy=strategy)
    """
    
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self._criterion = None
    
    def setup(self, context: TrainingContext):
        """Initialize loss criterion."""
        self._criterion = nn.CrossEntropyLoss()
    
    def compute_loss(self, outputs, labels, context):
        """Compute custom loss."""
        # Standard cross-entropy loss
        ce_loss = self._criterion(outputs, labels)
        
        # Add custom regularization term
        reg_term = self.alpha * torch.norm(outputs)
        
        return ce_loss + reg_term
```

### Step 3: Use Your Strategy

```python
from plato.trainers.composable import ComposableTrainer

trainer = ComposableTrainer(
    loss_strategy=MyCustomLossStrategy(alpha=0.3)
)
```

---

## API Reference

### TrainingContext

Shared context passed between strategies:

```python
class TrainingContext:
    model: nn.Module              # The model being trained
    device: torch.device          # CPU or GPU device
    client_id: int                # Client ID (0 for server)
    current_epoch: int            # Current epoch number
    current_round: int            # Current FL round number
    config: Dict[str, Any]        # Training configuration
    state: Dict[str, Any]         # Shared state between strategies
```

### Strategy Lifecycle

All strategies follow this lifecycle:

1. **Construction**: `strategy = MyStrategy(param=value)`
2. **Setup**: `strategy.setup(context)` - Called once at initialization
3. **Execution**: Strategy methods called during training
4. **Teardown**: `strategy.teardown(context)` - Called at end

### Common Patterns

#### Pattern 1: Access Model in Strategy

```python
def compute_loss(self, outputs, labels, context):
    model = context.model  # Access model
    device = context.device  # Access device
    # Use model and device...
```

#### Pattern 2: Share Data Between Strategies

```python
# Strategy 1: Store data
def on_train_start(self, context):
    context.state['my_data'] = some_value

# Strategy 2: Read data
def on_train_end(self, context):
    data = context.state.get('my_data')
```

#### Pattern 3: Combine Multiple Strategies

```python
composite = CompositeLossStrategy([
    (strategy1, weight1),
    (strategy2, weight2),
])
```

---

## Import Guide

### Import Base Interfaces

```python
from plato.trainers.strategies.base import (
    TrainingContext,
    LossCriterionStrategy,
    OptimizerStrategy,
    TrainingStepStrategy,
    LRSchedulerStrategy,
    ModelUpdateStrategy,
    DataLoaderStrategy,
)
```

### Import Default Implementations

```python
from plato.trainers.strategies import (
    CrossEntropyLossStrategy,
    AdamOptimizerStrategy,
    DefaultTrainingStepStrategy,
    CosineAnnealingLRSchedulerStrategy,
    NoOpUpdateStrategy,
    DefaultDataLoaderStrategy,
)
```

### Import Everything

```python
from plato.trainers.strategies import *
```

---

## Testing

### Unit Testing a Strategy

```python
import pytest
import torch
import torch.nn as nn
from plato.trainers.strategies.base import TrainingContext
from my_module import MyCustomLossStrategy

def test_my_custom_loss():
    # Create strategy
    strategy = MyCustomLossStrategy(alpha=0.5)
    
    # Create context
    context = TrainingContext()
    context.model = nn.Linear(10, 2)
    context.device = torch.device('cpu')
    
    # Setup strategy
    strategy.setup(context)
    
    # Test loss computation
    outputs = torch.randn(10, 2)
    labels = torch.randint(0, 2, (10,))
    
    loss = strategy.compute_loss(outputs, labels, context)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert loss.item() > 0
```

---

## FAQ

**Q: When should I use strategies vs callbacks?**
A: Use strategies for algorithmic variations (loss, optimizer, training step). Use callbacks for event-driven behavior (logging, checkpointing).

**Q: Can I use multiple strategies of the same type?**
A: Use `CompositeLossStrategy` or `CompositeUpdateStrategy` to combine multiple strategies.

**Q: How do I migrate from inheritance?**
A: See `TRAINER_REFACTORING_EXAMPLES.md` for migration patterns and examples.

**Q: Are strategies compatible with existing trainers?**
A: Phase 1 is complete. Phase 2 (ComposableTrainer) is needed to use strategies in training.

**Q: Can strategies access the training loop?**
A: Strategies receive a `TrainingContext` with model, device, config, and shared state.

---

## Status

- ✅ **Phase 1 Complete**: Strategy interfaces and default implementations
- ⏳ **Phase 2 Pending**: ComposableTrainer implementation
- ⏳ **Phase 3 Pending**: Algorithm-specific strategies (FedProx, SCAFFOLD, etc.)

---

## Contributing

To add a new strategy:

1. Choose the appropriate strategy type
2. Inherit from the base class
3. Implement required abstract methods
4. Add comprehensive docstrings with examples
5. Add type hints to all methods
6. Write unit tests
7. Submit pull request

---

## Resources

- **Design Documents**: See `TRAINER_REFACTORING_*.md` files in root
- **Examples**: Check docstrings in each strategy class
- **Tests**: `tests/trainers/strategies/`
- **Source Code**: `plato/trainers/strategies/`

---

**Version**: 1.0 (Phase 1)  
**Last Updated**: 2024  
**Status**: Production Ready (Interfaces Complete)