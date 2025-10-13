# Trainers

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [ComposableTrainer API](#composabletrainer-api)
5. [Strategy Interfaces](#strategy-interfaces)
6. [Default Strategies](#default-strategies)
7. [Algorithm-Specific Strategies](#algorithm-specific-strategies)
8. [TrainingContext](#trainingcontext)
9. [Creating Custom Strategies](#creating-custom-strategies)
10. [Advanced Usage of Strategies](#advanced-usage-of-strategies)
11. [Usage Examples](#usage-examples)
12. [API Reference](#api-reference)
13. [Common Patterns](#common-patterns)
14. [Customizing Trainers using Callbacks](#customizing-trainers-using-callbacks)
15. [Accessing and Customizing the Run History During Training](#accessing-and-customizing-the-run-history-during-training)
16. [Customizing Trainers using Subclassing and Hooks](#customizing-trainers-using-subclassing-and-hooks)
17. [Import Guide](#import-guide)
18. [Frequently Asked Questions](#frequently-asked-questions)

---

## Overview

Plato's trainer system uses a **composition-based architecture** built on the **Strategy Pattern** and **Dependency Injection**. Instead of creating subclasses that override methods, you inject strategy objects that define specific behaviors.

### Why Strategies?

- **Composability**: Mix and match strategies to create new algorithms
- **Testability**: Test strategies in isolation without full trainer setup
- **Flexibility**: Swap strategies at runtime or configuration time
- **Maintainability**: Fix bugs once in strategies, benefit everywhere
- **Clarity**: Each strategy has a single, clear responsibility

---

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────┐
│            ComposableTrainer                        │
│  ┌───────────────────────────────────────────────┐  │
│  │        TrainingContext                        │  │
│  │  (Shared state between strategies)            │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  LossCriterionStrategy                        │  │
│  │  • compute_loss(outputs, labels, context)     │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  OptimizerStrategy                            │  │
│  │  • create_optimizer(model, context)           │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  TrainingStepStrategy                         │  │
│  │  • training_step(model, opt, data, context)   │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  LRSchedulerStrategy                          │  │
│  │  • create_scheduler(optimizer, context)       │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  ModelUpdateStrategy                          │  │
│  │  • on_train_start/end, before/after_step      │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  DataLoaderStrategy                           │  │
│  │  • create_train_loader(dataset, context)      │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  TestingStrategy                              │  │
│  │  • test_model(model, testset, context)        │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Strategy Types

| Strategy | Purpose | When to Customize |
|----------|---------|-------------------|
| **LossCriterionStrategy** | Compute loss | Custom loss functions, regularization (FedProx, FedDyn) |
| **OptimizerStrategy** | Create optimizer | Custom optimizers, parameter groups (FedMos) |
| **TrainingStepStrategy** | Training logic | Multiple passes, gradient accumulation (LG-FedAvg, APFL) |
| **LRSchedulerStrategy** | LR scheduling | Custom schedules, warmup |
| **ModelUpdateStrategy** | State management | Control variates, personalization (SCAFFOLD, Ditto) |
| **DataLoaderStrategy** | Data loading | Custom sampling, augmentation |

---

## Quick Start

### Example 1: Default Trainer

```python
from plato.trainers.composable import ComposableTrainer

# Uses sensible defaults for everything
trainer = ComposableTrainer(model=my_model)
```

### Example 2: FedProx Trainer

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import FedProxLossStrategy

# FedProx with proximal term regularization
trainer = ComposableTrainer(
    model=my_model,
    loss_strategy=FedProxLossStrategy(mu=0.01)
)
```

### Example 3: Custom Optimizer

```python
from plato.trainers.strategies import AdamOptimizerStrategy

trainer = ComposableTrainer(
    model=my_model,
    optimizer_strategy=AdamOptimizerStrategy(lr=0.001, weight_decay=0.01)
)
```

### Example 4: Combining Strategies

```python
from plato.trainers.strategies.algorithms import (
    FedProxLossStrategy,
    LGFedAvgStepStrategy,
)

# FedProx loss + LG-FedAvg training
trainer = ComposableTrainer(
    model=my_model,
    loss_strategy=FedProxLossStrategy(mu=0.01),
    training_step_strategy=LGFedAvgStepStrategy(
        global_layer_names=['conv1', 'conv2', 'fc1'],
        local_layer_names=['fc2']
    )
)
```

---

## ComposableTrainer

### Class: `ComposableTrainer`

**Location**: `plato.trainers.composable`

```python
class ComposableTrainer(base.Trainer):
    """Composable trainer using strategies for extensibility."""

    def __init__(
        self,
        model: Optional[Union[nn.Module, Callable[[], nn.Module]]] = None,
        callbacks: Optional[List[Any]] = None,
        loss_strategy: Optional[LossCriterionStrategy] = None,
        optimizer_strategy: Optional[OptimizerStrategy] = None,
        training_step_strategy: Optional[TrainingStepStrategy] = None,
        lr_scheduler_strategy: Optional[LRSchedulerStrategy] = None,
        model_update_strategy: Optional[ModelUpdateStrategy] = None,
        data_loader_strategy: Optional[DataLoaderStrategy] = None,
        testing_strategy: Optional[TestingStrategy] = None,
    )
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` or callable | `None` | Model instance or factory function. If None, uses model from registry |
| `callbacks` | `list` | `None` | List of callback classes or instances |
| `loss_strategy` | `LossCriterionStrategy` | `DefaultLossCriterionStrategy()` | Strategy for computing loss |
| `optimizer_strategy` | `OptimizerStrategy` | `DefaultOptimizerStrategy()` | Strategy for creating optimizer |
| `training_step_strategy` | `TrainingStepStrategy` | `DefaultTrainingStepStrategy()` | Strategy for training step logic |
| `lr_scheduler_strategy` | `LRSchedulerStrategy` | `DefaultLRSchedulerStrategy()` | Strategy for LR scheduling |
| `model_update_strategy` | `ModelUpdateStrategy` | `NoOpUpdateStrategy()` | Strategy for state management |
| `data_loader_strategy` | `DataLoaderStrategy` | `DefaultDataLoaderStrategy()` | Strategy for data loading |
| `testing_strategy` | `TestingStrategy` | `DefaultTestingStrategy()` | Strategy for model evaluation |

#### Key Methods

!!! note "`train(trainset, sampler, **kwargs) -> float`"

    Train the model on the given dataset and sampler.

    **Parameters:**

    - `trainset`: Training dataset
    - `sampler`: Data sampler for this client
    - `**kwargs`: Additional arguments

    **Returns:**

    - Training time in seconds

    **Example:**

    ```python
    training_time = trainer.train(trainset, sampler)
    ```

!!! note "`test(testset, sampler=None, **kwargs) -> float`"
    Test the model on the given dataset.

    **Parameters:**

    - `testset`: Test dataset
    - `sampler`: Optional data sampler
    - `**kwargs`: Additional arguments

    **Returns:**

    - Test accuracy (0.0 to 1.0)

    **Example:**

    ```python
    accuracy = trainer.test(testset)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    ```

!!! note "`train_model(config, trainset, sampler, **kwargs)`"
    Main training loop implementation. Called internally by `train()`.

    **Parameters:**

    - `config`: Configuration dictionary
    - `trainset`: Training dataset
    - `sampler`: Data sampler
    - `**kwargs`: Additional arguments

!!! note "`save_model(filename=None, location=None)`"
    Save model weights and training history.

    **Parameters:**

    - `filename`: Optional custom filename
    - `location`: Optional custom directory

    **Example:**

    ```python
    trainer.save_model("my_model.pth")
    ```

!!! note "`load_model(filename=None, location=None)`"
    Load model weights and training history.

    **Parameters:**

    - `filename`: Optional custom filename
    - `location`: Optional custom directory

    **Example:**

    ```python
    trainer.load_model("my_model.pth")
    ```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | The neural network model |
| `context` | `TrainingContext` | Shared state between strategies |
| `optimizer` | `torch.optim.Optimizer` | Current optimizer (set during training) |
| `lr_scheduler` | `_LRScheduler` | Current LR scheduler (set during training) |
| `train_loader` | `DataLoader` | Current training data loader |
| `device` | `torch.device` | Device for training (CPU/GPU) |
| `client_id` | `int` | Client ID (0 for server) |
| `current_epoch` | `int` | Current training epoch (1-indexed) |
| `current_round` | `int` | Current FL round (1-indexed) |
| `run_history` | `RunHistory` | Training metrics history |
| `accuracy` | `float` | Latest test accuracy |

---

## Basic Usage

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

## Strategy Interfaces

### Base: `Strategy`

**Location**: `plato.trainers.strategies.base`

All strategies inherit from this base class.

```python
class Strategy(ABC):
    """Base class for all strategies."""

    def setup(self, context: TrainingContext) -> None:
        """Called once during trainer initialization."""
        pass

    def teardown(self, context: TrainingContext) -> None:
        """Called when all training is complete."""
        pass
```

#### Lifecycle Hooks

| Method | When Called | Purpose |
|--------|-------------|---------|
| `setup(context)` | Once at trainer initialization | Initialize state, access model architecture |
| `teardown(context)` | Once when training complete | Clean up resources, save final state |

---

### LossCriterionStrategy

**Location**: `plato.trainers.strategies.base`

Strategy for computing loss during training.

```python
class LossCriterionStrategy(Strategy):
    """Strategy interface for computing loss."""

    @abstractmethod
    def compute_loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        context: TrainingContext
    ) -> torch.Tensor:
        """
        Compute loss given model outputs and labels.

        Args:
            outputs: Model predictions (e.g., logits)
            labels: Ground truth labels
            context: Training context

        Returns:
            Scalar loss tensor
        """
        pass
```

#### When to Implement

- Custom loss functions (contrastive, triplet, etc.)
- Adding regularization terms (L2, proximal, etc.)
- Multi-task losses
- Adaptive loss weighting

#### Example Implementation

```python
from plato.trainers.strategies.base import LossCriterionStrategy, TrainingContext
import torch.nn as nn

class MyLossStrategy(LossCriterionStrategy):
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self._criterion = None

    def setup(self, context: TrainingContext):
        self._criterion = nn.CrossEntropyLoss()

    def compute_loss(self, outputs, labels, context):
        base_loss = self._criterion(outputs, labels)
        reg_term = self.alpha * torch.norm(outputs)
        return base_loss + reg_term
```

---

### OptimizerStrategy

**Location**: `plato.trainers.strategies.base`

Strategy for creating and configuring optimizers.

```python
class OptimizerStrategy(Strategy):
    """Strategy interface for creating optimizers."""

    @abstractmethod
    def create_optimizer(
        self,
        model: nn.Module,
        context: TrainingContext
    ) -> torch.optim.Optimizer:
        """
        Create and return optimizer for training.

        Args:
            model: The model to optimize
            context: Training context

        Returns:
            Configured optimizer instance
        """
        pass

    def on_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        context: TrainingContext
    ) -> None:
        """
        Hook called after optimizer.step().

        Args:
            optimizer: The optimizer that just stepped
            context: Training context
        """
        pass
```

#### When to Implement

- Custom optimizer type (Adam, SGD, etc.)
- Parameter groups with different learning rates
- Custom optimizer implementations (FedMos)
- Gradient clipping or other post-step processing

#### Example Implementation

```python
from plato.trainers.strategies.base import OptimizerStrategy
import torch.optim as optim

class MyOptimizerStrategy(OptimizerStrategy):
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum

    def create_optimizer(self, model, context):
        return optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum
        )
```

---

### TrainingStepStrategy

**Location**: `plato.trainers.strategies.base`

Strategy for performing forward and backward passes.

```python
class TrainingStepStrategy(Strategy):
    """Strategy interface for training step logic."""

    @abstractmethod
    def training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        examples: torch.Tensor,
        labels: torch.Tensor,
        loss_criterion: callable,
        context: TrainingContext,
    ) -> torch.Tensor:
        """
        Perform one training step.

        Args:
            model: The model to train
            optimizer: The optimizer
            examples: Input batch (on device)
            labels: Target labels (on device)
            loss_criterion: Callable that computes loss
            context: Training context

        Returns:
            Loss value for this step
        """
        pass
```

#### When to Implement

- Multiple forward/backward passes (LG-FedAvg, APFL)
- Gradient accumulation
- Mixed precision training
- Gradient clipping
- Custom training loops

#### Example Implementation

```python
from plato.trainers.strategies.base import TrainingStepStrategy

class MyStepStrategy(TrainingStepStrategy):
    def training_step(self, model, optimizer, examples, labels,
                     loss_criterion, context):
        optimizer.zero_grad()

        outputs = model(examples)
        loss = loss_criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        return loss
```

---

### LRSchedulerStrategy

**Location**: `plato.trainers.strategies.base`

Strategy for learning rate scheduling.

```python
class LRSchedulerStrategy(Strategy):
    """Strategy interface for learning rate scheduling."""

    @abstractmethod
    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        context: TrainingContext
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create and return LR scheduler.

        Args:
            optimizer: The optimizer to schedule
            context: Training context

        Returns:
            LR scheduler instance, or None for no scheduling
        """
        pass

    def step(
        self,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        context: TrainingContext,
    ) -> None:
        """
        Perform one scheduler step.

        Args:
            scheduler: The scheduler (may be None)
            context: Training context
        """
        if scheduler is not None:
            scheduler.step()
```

#### When to Implement

- Step decay scheduling
- Cosine annealing
- Warmup schedules
- Custom adaptive schedules

#### Example Implementation

```python
from plato.trainers.strategies.base import LRSchedulerStrategy
import torch.optim.lr_scheduler as lr_scheduler

class MySchedulerStrategy(LRSchedulerStrategy):
    def __init__(self, step_size=10, gamma=0.1):
        self.step_size = step_size
        self.gamma = gamma

    def create_scheduler(self, optimizer, context):
        return lr_scheduler.StepLR(
            optimizer,
            step_size=self.step_size,
            gamma=self.gamma
        )
```

---

### ModelUpdateStrategy

**Location**: `plato.trainers.strategies.base`

Strategy for managing model updates and state.

```python
class ModelUpdateStrategy(Strategy):
    """Strategy interface for model updates and state management."""

    def on_train_start(self, context: TrainingContext) -> None:
        """Called at the start of each training round."""
        pass

    def on_train_end(self, context: TrainingContext) -> None:
        """Called at the end of training round."""
        pass

    def before_step(self, context: TrainingContext) -> None:
        """Called before each training step."""
        pass

    def after_step(self, context: TrainingContext) -> None:
        """Called after each training step."""
        pass

    def get_update_payload(self, context: TrainingContext) -> Dict[str, Any]:
        """Return additional data to send to server."""
        return {}
```

#### Lifecycle Methods

| Method | When Called | Common Uses |
|--------|-------------|-------------|
| `on_train_start(context)` | Start of each round | Receive server data, save global weights |
| `on_train_end(context)` | End of each round | Compute state updates, save to disk |
| `before_step(context)` | Before each step | Pre-step modifications |
| `after_step(context)` | After each step | Apply corrections (SCAFFOLD), accumulate state |
| `get_update_payload(context)` | After training | Return data for server (control variates, etc.) |

#### When to Implement

- Control variates (SCAFFOLD)
- Dynamic regularization state (FedDyn)
- Personalization (FedPer, FedRep, Ditto)
- Layer freezing/unfreezing
- Custom state management

#### Example Implementation

```python
from plato.trainers.strategies.base import ModelUpdateStrategy
import copy

class MyUpdateStrategy(ModelUpdateStrategy):
    def __init__(self):
        self.global_weights = None

    def on_train_start(self, context):
        # Save global model weights
        self.global_weights = copy.deepcopy(context.model.state_dict())

    def after_step(self, context):
        # Apply custom corrections after each step
        with torch.no_grad():
            for name, param in context.model.named_parameters():
                # Your custom logic here
                pass
```

---

### TestingStrategy

**Location**: `plato.trainers.strategies.base`

Strategy for testing/evaluating model performance.

```python
class TestingStrategy(Strategy):
    """Strategy interface for model testing and evaluation."""

    @abstractmethod
    def test_model(
        self,
        model: torch.nn.Module,
        config: dict,
        testset: Dataset,
        sampler: Optional[Sampler],
        context: TrainingContext
    ) -> float:
        """
        Test the model using evaluation metrics.

        Args:
            model: The model to test
            config: Testing configuration dictionary
            testset: Test dataset
            sampler: Optional data sampler for test set
            context: Training context

        Returns:
            Evaluation metric (e.g., accuracy) as a float
        """
        pass
```

#### When to Implement

- Custom evaluation metrics
- Multi-metric evaluation
- Task-specific testing procedures
- Specialized model evaluation flows

#### Example Implementation

```python
class DefaultTestingStrategy(TestingStrategy):
    """Default testing strategy for classification tasks."""

    def test_model(self, model, config, testset, sampler, context):
        model.eval()
        test_loader = DataLoader(testset, batch_size=32, sampler=sampler)

        correct = 0
        total = 0

        with torch.no_grad():
            for examples, labels in test_loader:
                examples = examples.to(context.device)
                labels = labels.to(context.device)

                outputs = model(examples)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
```

### DataLoaderStrategy

**Location**: `plato.trainers.strategies.base`

Strategy for creating data loaders.

```python
class DataLoaderStrategy(Strategy):
    """Strategy interface for creating data loaders."""

    @abstractmethod
    def create_train_loader(
        self,
        trainset,
        sampler,
        batch_size: int,
        context: TrainingContext
    ) -> torch.utils.data.DataLoader:
        """
        Create training data loader.

        Args:
            trainset: Training dataset
            sampler: Data sampler
            batch_size: Batch size
            context: Training context

        Returns:
            Configured DataLoader instance
        """
        pass
```

#### When to Implement

- Custom sampling strategies
- Data augmentation
- Prefetching
- Custom collate functions
- Distributed data loading

#### Example Implementation

```python
from plato.trainers.strategies.base import DataLoaderStrategy
import torch.utils.data as data

class MyDataLoaderStrategy(DataLoaderStrategy):
    def __init__(self, num_workers=4, prefetch_factor=2):
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    def create_train_loader(self, trainset, sampler, batch_size, context):
        return data.DataLoader(
            trainset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True
        )
```

---

## Default Strategies

### Loss Criterion Strategies

**Location**: `plato.trainers.strategies.loss_criterion`

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `DefaultLossCriterionStrategy` | Uses framework's loss registry | Default behavior |
| `CrossEntropyLossStrategy` | Cross-entropy loss | Classification |
| `MSELossStrategy` | Mean squared error | Regression |
| `BCEWithLogitsLossStrategy` | Binary cross-entropy | Binary classification |
| `NLLLossStrategy` | Negative log likelihood | After log-softmax |
| `L1LossStrategy` | L1 loss | Regression with outliers |
| `SmoothL1LossStrategy` | Smooth L1 loss | Robust regression |

**Example:**
```python
from plato.trainers.strategies import CrossEntropyLossStrategy

trainer = ComposableTrainer(
    loss_strategy=CrossEntropyLossStrategy(label_smoothing=0.1)
)
```

### Optimizer Strategies

**Location**: `plato.trainers.strategies.optimizer`

| Strategy | Description | Parameters |
|----------|-------------|------------|
| `DefaultOptimizerStrategy` | Uses framework's optimizer | From config |
| `SGDOptimizerStrategy` | Stochastic gradient descent | `lr`, `momentum`, `weight_decay` |
| `AdamOptimizerStrategy` | Adam optimizer | `lr`, `betas`, `weight_decay` |
| `AdamWOptimizerStrategy` | Adam with weight decay | `lr`, `betas`, `weight_decay` |
| `RMSpropOptimizerStrategy` | RMSprop optimizer | `lr`, `alpha`, `momentum` |
| `AdagradOptimizerStrategy` | Adagrad optimizer | `lr`, `lr_decay`, `weight_decay` |

**Example:**
```python
from plato.trainers.strategies import AdamOptimizerStrategy

trainer = ComposableTrainer(
    optimizer_strategy=AdamOptimizerStrategy(lr=0.001, weight_decay=0.01)
)
```

### Training Step Strategies

**Location**: `plato.trainers.strategies.training_step`

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `DefaultTrainingStepStrategy` | Standard forward-backward-step | Default training |
| `GradientAccumulationStepStrategy` | Accumulate gradients over steps | Large effective batch size |
| `MixedPrecisionStepStrategy` | FP16 training with GradScaler | Speed up training on GPU |
| `GradientClippingStepStrategy` | Clip gradients by norm | Prevent gradient explosion |

**Example:**
```python
from plato.trainers.strategies import GradientAccumulationStepStrategy

trainer = ComposableTrainer(
    training_step_strategy=GradientAccumulationStepStrategy(
        accumulation_steps=4  # 4x effective batch size
    )
)
```

### LR Scheduler Strategies

**Location**: `plato.trainers.strategies.lr_scheduler`

| Strategy | Description | Parameters |
|----------|-------------|------------|
| `DefaultLRSchedulerStrategy` | Uses framework's scheduler | From config |
| `StepLRSchedulerStrategy` | Step decay | `step_size`, `gamma` |
| `MultiStepLRSchedulerStrategy` | Multi-step decay | `milestones`, `gamma` |
| `ExponentialLRSchedulerStrategy` | Exponential decay | `gamma` |
| `CosineAnnealingLRSchedulerStrategy` | Cosine annealing | `T_max`, `eta_min` |
| `ReduceLROnPlateauSchedulerStrategy` | Reduce on plateau | `mode`, `factor`, `patience` |
| `WarmupSchedulerStrategy` | Linear warmup | `warmup_epochs`, `base_scheduler` |

**Example:**
```python
from plato.trainers.strategies import CosineAnnealingLRSchedulerStrategy

trainer = ComposableTrainer(
    lr_scheduler_strategy=CosineAnnealingLRSchedulerStrategy(T_max=50)
)
```

### Model Update Strategies

**Location**: `plato.trainers.strategies.model_update`

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `NoOpUpdateStrategy` | No additional updates | Default (no state management) |
| `StateTrackingUpdateStrategy` | Track arbitrary state | Custom state management |

**Example:**
```python
from plato.trainers.strategies import NoOpUpdateStrategy

trainer = ComposableTrainer(
    model_update_strategy=NoOpUpdateStrategy()
)
```

### Data Loader Strategies

**Location**: `plato.trainers.strategies.data_loader`

| Strategy | Description | Parameters |
|----------|-------------|------------|
| `DefaultDataLoaderStrategy` | Standard DataLoader | Uses config settings |
| `PrefetchDataLoaderStrategy` | With prefetching | `num_workers`, `prefetch_factor` |
| `PinMemoryDataLoaderStrategy` | Pin memory for GPU | Faster GPU transfer |

**Example:**
```python
from plato.trainers.strategies import DefaultDataLoaderStrategy

trainer = ComposableTrainer(
    data_loader_strategy=DefaultDataLoaderStrategy()
)
```

---

## Algorithm-Specific Strategies

### FedProx

**Location**: `plato.trainers.strategies.algorithms.fedprox_strategy`

**Reference**: Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020.

#### Strategies

**`FedProxLossStrategy`**

Adds proximal term to loss: `loss = base_loss + (μ/2) * ||w - w_global||²`

```python
class FedProxLossStrategy(LossCriterionStrategy):
    def __init__(self, mu=0.01, base_loss_fn=None, norm_type='l2'):
        """
        Args:
            mu: Proximal term coefficient (default: 0.01)
            base_loss_fn: Base loss function (default: CrossEntropyLoss)
            norm_type: 'l2' or 'l1' (default: 'l2')
        """
```

**`FedProxLossStrategyFromConfig`**

Reads `mu` from config: `clients.proximal_term_penalty_constant`

#### Usage

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import FedProxLossStrategy

# Explicit parameters
trainer = ComposableTrainer(
    loss_strategy=FedProxLossStrategy(mu=0.01)
)

# From config
from plato.trainers.strategies.algorithms import FedProxLossStrategyFromConfig
trainer = ComposableTrainer(
    loss_strategy=FedProxLossStrategyFromConfig()
)
```

#### Config Example

```yaml
clients:
  proximal_term_penalty_constant: 0.01
```

---

### SCAFFOLD

**Location**: `plato.trainers.strategies.algorithms.scaffold_strategy`

**Reference**: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning", ICML 2020.

#### Strategies

**`SCAFFOLDUpdateStrategy`**

Manages control variates to reduce client drift.

```python
class SCAFFOLDUpdateStrategy(ModelUpdateStrategy):
    def __init__(self, save_path=None):
        """
        Args:
            save_path: Custom path for saving control variates (optional)
        """
```

**`SCAFFOLDUpdateStrategyV2`**

Alternative implementation (Option 1 from paper).

#### Usage

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import SCAFFOLDUpdateStrategy

trainer = ComposableTrainer(
    model_update_strategy=SCAFFOLDUpdateStrategy()
)

# In your client code, pass server control variate
trainer.context.state['server_control_variate'] = server_cv

# Train
trainer.train(trainset, sampler)

# Get control variate delta for server
payload = trainer.model_update_strategy.get_update_payload(trainer.context)
delta = payload['control_variate_delta']
```

#### Server-Side Integration

```python
# Server aggregates control variate deltas
for client_id, delta in client_deltas.items():
    for name in server_control_variate:
        server_control_variate[name] += (1/K) * delta[name]

# Send updated server control variate to clients
trainer.context.state['server_control_variate'] = server_control_variate
```

---

### FedDyn

**Location**: `plato.trainers.strategies.algorithms.feddyn_strategy`

**Reference**: Acar et al., "Federated Learning Based on Dynamic Regularization", ICLR 2021.

#### Strategies

**`FedDynLossStrategy`**

Adds dynamic regularization term to loss.

```python
class FedDynLossStrategy(LossCriterionStrategy):
    def __init__(self, alpha=0.01, adaptive_alpha=True, base_loss_fn=None):
        """
        Args:
            alpha: Regularization coefficient (default: 0.01)
            adaptive_alpha: Scale by client data weight (default: True)
            base_loss_fn: Base loss function (default: CrossEntropyLoss)
        """
```

**`FedDynUpdateStrategy`**

Manages FedDyn state (gradient accumulator).

```python
class FedDynUpdateStrategy(ModelUpdateStrategy):
    """Manages FedDyn state."""
```

**`FedDynLossStrategyFromConfig`**

Reads `alpha` from config: `algorithm.alpha_coef`

#### Usage

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import (
    FedDynLossStrategy,
    FedDynUpdateStrategy
)

# Must use both strategies together
trainer = ComposableTrainer(
    loss_strategy=FedDynLossStrategy(alpha=0.01),
    model_update_strategy=FedDynUpdateStrategy()
)
```

#### Config Example

```yaml
algorithm:
  alpha_coef: 0.01
```

---

### LG-FedAvg

**Location**: `plato.trainers.strategies.algorithms.lgfedavg_strategy`

**Reference**: Liang et al., "Think Locally, Act Globally: Federated Learning with Local and Global Representations", 2020.

#### Strategies

**`LGFedAvgStepStrategy`**

Performs dual forward/backward passes for local and global layers.

```python
class LGFedAvgStepStrategy(TrainingStepStrategy):
    def __init__(self, global_layer_names, local_layer_names,
                 train_local_first=True):
        """
        Args:
            global_layer_names: Layer name patterns for global layers
            local_layer_names: Layer name patterns for local layers
            train_local_first: Train local layers first (default: True)
        """
```

**`LGFedAvgStepStrategyFromConfig`**

Reads layer names from config: `algorithm.global_layer_names`, `algorithm.local_layer_names`

**`LGFedAvgStepStrategyAuto`**

Automatically treats last N layers as local.

```python
class LGFedAvgStepStrategyAuto(TrainingStepStrategy):
    def __init__(self, num_local_layers=1, train_local_first=True):
        """
        Args:
            num_local_layers: Number of final layers to treat as local
            train_local_first: Train local layers first (default: True)
        """
```

#### Usage

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import LGFedAvgStepStrategy

# Explicit layer names
trainer = ComposableTrainer(
    training_step_strategy=LGFedAvgStepStrategy(
        global_layer_names=['conv1', 'conv2', 'fc1'],
        local_layer_names=['fc2']
    )
)

# Auto detection
from plato.trainers.strategies.algorithms import LGFedAvgStepStrategyAuto
trainer = ComposableTrainer(
    training_step_strategy=LGFedAvgStepStrategyAuto(num_local_layers=1)
)
```

#### Config Example

```yaml
algorithm:
  global_layer_names:
    - conv1
    - conv2
    - fc1
  local_layer_names:
    - fc2
  train_local_first: true
```

---

### FedMos

**Location**: `plato.trainers.strategies.algorithms.fedmos_strategy`

**Reference**: Wang et al., "FedMos: Taming Client Drift in Federated Learning with Double Momentum", IEEE INFOCOM 2023.

#### Strategies

**`FedMosOptimizerStrategy`**

Custom optimizer with local and global momentum.

```python
class FedMosOptimizerStrategy(OptimizerStrategy):
    def __init__(self, lr=0.01, a=0.9, mu=0.9, weight_decay=0):
        """
        Args:
            lr: Learning rate (default: 0.01)
            a: Local momentum coefficient (default: 0.9)
            mu: Global momentum coefficient (default: 0.9)
            weight_decay: Weight decay (default: 0)
        """
```

**`FedMosUpdateStrategy`**

Manages global momentum state.

**`FedMosOptimizerStrategyFromConfig`**

Reads parameters from config: `algorithm.a`, `algorithm.mu`, `parameters.optimizer.lr`

#### Usage

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import (
    FedMosOptimizerStrategy,
    FedMosUpdateStrategy
)

# Must use both strategies together
trainer = ComposableTrainer(
    optimizer_strategy=FedMosOptimizerStrategy(lr=0.01, a=0.9, mu=0.9),
    model_update_strategy=FedMosUpdateStrategy()
)
```

#### Config Example

```yaml
algorithm:
  a: 0.9
  mu: 0.9
parameters:
  optimizer:
    lr: 0.01
```

---

### FedPer

**Location**: `plato.trainers.strategies.algorithms.personalized_fl_strategy`

**Reference**: Arivazhagan et al., "Federated Learning with Personalization Layers", 2019.

#### Strategies

**`FedPerUpdateStrategy`**

Freezes global layers during personalization phase.

```python
class FedPerUpdateStrategy(ModelUpdateStrategy):
    def __init__(self, global_layer_names, personalization_rounds=None):
        """
        Args:
            global_layer_names: Layer name patterns for global layers
            personalization_rounds: Rounds for personalization (optional)
        """
```

**`FedPerUpdateStrategyFromConfig`**

Reads from config: `algorithm.global_layer_names`, `trainer.rounds`

#### Usage

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import FedPerUpdateStrategy

trainer = ComposableTrainer(
    model_update_strategy=FedPerUpdateStrategy(
        global_layer_names=['conv1', 'conv2', 'fc1'],
        personalization_rounds=20
    )
)
```

#### Config Example

```yaml
algorithm:
  global_layer_names:
    - conv1
    - conv2
    - fc1
trainer:
  rounds: 100  # Personalization starts after this
```

---

### FedRep

**Location**: `plato.trainers.strategies.algorithms.personalized_fl_strategy`

**Reference**: Collins et al., "Exploiting Shared Representations for Personalized Federated Learning", ICML 2021.

#### Strategies

**`FedRepUpdateStrategy`**

Alternates between training local and global layers.

```python
class FedRepUpdateStrategy(ModelUpdateStrategy):
    def __init__(self, global_layer_names, local_layer_names, local_epochs=1):
        """
        Args:
            global_layer_names: Layer name patterns for global layers
            local_layer_names: Layer name patterns for local layers
            local_epochs: Epochs for local layer training (default: 1)
        """
```

**`FedRepUpdateStrategyFromConfig`**

Reads from config: `algorithm.global_layer_names`, `algorithm.local_layer_names`, `algorithm.local_epochs`

#### Usage

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import FedRepUpdateStrategy

trainer = ComposableTrainer(
    model_update_strategy=FedRepUpdateStrategy(
        global_layer_names=['conv1', 'conv2'],
        local_layer_names=['fc1', 'fc2'],
        local_epochs=2
    )
)
```

#### Training Flow

1. Train local layers for `local_epochs` epochs
2. Train global layers for remaining epochs
3. Repeat each round

#### Config Example

```yaml
algorithm:
  global_layer_names:
    - conv1
    - conv2
  local_layer_names:
    - fc1
    - fc2
  local_epochs: 2
```

---

### APFL

**Location**: `plato.trainers.strategies.algorithms.apfl_strategy`

**Reference**: Deng et al., "Adaptive Personalized Federated Learning", 2020.

#### Strategies

**`APFLUpdateStrategy`**

Manages global and personalized models with mixing parameter α.

```python
class APFLUpdateStrategy(ModelUpdateStrategy):
    def __init__(self, alpha=0.5, adaptive_alpha=True, alpha_lr=0.01,
                 model_fn=None):
        """
        Args:
            alpha: Initial mixing parameter (0=fully personalized, 1=fully global)
            adaptive_alpha: Learn α adaptively (default: True)
            alpha_lr: Learning rate for α (default: 0.01)
            model_fn: Function to create personalized model (optional)
        """
```

**`APFLStepStrategy`**

Performs dual model training step.

**`APFLUpdateStrategyFromConfig`**

Reads from config: `algorithm.alpha`, `algorithm.adaptive_alpha`, `algorithm.alpha_lr`

#### Usage

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import (
    APFLUpdateStrategy,
    APFLStepStrategy
)

# Must use both strategies together
trainer = ComposableTrainer(
    model_update_strategy=APFLUpdateStrategy(alpha=0.5),
    training_step_strategy=APFLStepStrategy()
)
```

#### How It Works

- Maintains global model (w) and personalized model (v)
- Output = α * v + (1 - α) * w
- α is learned adaptively per client based on validation loss

#### Config Example

```yaml
algorithm:
  alpha: 0.5
  adaptive_alpha: true
  alpha_lr: 0.01
```

---

### Ditto

**Location**: `plato.trainers.strategies.algorithms.ditto_strategy`

**Reference**: Li et al., "Ditto: Fair and Robust Federated Learning Through Personalization", ICML 2021.

#### Strategies

**`DittoUpdateStrategy`**

Post-training personalization with regularization towards global model.

```python
class DittoUpdateStrategy(ModelUpdateStrategy):
    def __init__(self, ditto_lambda=0.1, personalization_epochs=5,
                 model_fn=None):
        """
        Args:
            ditto_lambda: Regularization coefficient (default: 0.1)
            personalization_epochs: Epochs for personalization (default: 5)
            model_fn: Function to create personalized model (optional)
        """
```

**`DittoUpdateStrategyFromConfig`**

Reads from config: `algorithm.ditto_lambda`, `algorithm.personalization_epochs`

#### Usage

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import DittoUpdateStrategy

trainer = ComposableTrainer(
    model_update_strategy=DittoUpdateStrategy(
        ditto_lambda=0.1,
        personalization_epochs=5
    )
)
```

#### Training Flow

1. Train global model normally
2. After global training, train personalized model
3. Personalized model regularized: min F(v) + λ/2 ||v - w||²
4. Send only global model to server
5. Use personalized model for inference

#### Config Example

```yaml
algorithm:
  ditto_lambda: 0.1
  personalization_epochs: 5
```

---

## TrainingContext

**Location**: `plato.trainers.strategies.base`

### Class: `TrainingContext`

Shared state container passed between strategies.

```python
class TrainingContext:
    """Shared context passed between strategies."""

    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.device: Optional[torch.device] = None
        self.client_id: int = 0
        self.current_epoch: int = 0
        self.current_round: int = 0
        self.config: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | The model being trained |
| `device` | `torch.device` | CPU or GPU device |
| `client_id` | `int` | Client identifier (0 for server) |
| `current_epoch` | `int` | Current training epoch (1-indexed) |
| `current_round` | `int` | Current FL round (1-indexed) |
| `config` | `Dict[str, Any]` | Training configuration |
| `state` | `Dict[str, Any]` | Shared state between strategies |

### Using `context.state`

The `state` dictionary allows strategies to communicate:

```python
# Producer strategy
class ProducerStrategy(ModelUpdateStrategy):
    def on_train_start(self, context):
        # Store data for other strategies
        context.state['shared_data'] = compute_data()

# Consumer strategy
class ConsumerStrategy(LossCriterionStrategy):
    def compute_loss(self, outputs, labels, context):
        # Retrieve data from context
        data = context.state.get('shared_data')
        return compute_loss_with(data)
```

### Standard `context.state` Keys

| Key | Type | Set By | Used By | Description |
|-----|------|--------|---------|-------------|
| `server_control_variate` | `Dict` | Server/Client | SCAFFOLD | Server control variate |
| `client_control_variate_delta` | `Dict` | SCAFFOLD | Server | Control variate delta |
| `train_loader` | `DataLoader` | Trainer | Strategies | Current data loader |
| `current_batch` | `int` | Trainer | Strategies | Current batch ID |
| `last_loss` | `float` | Trainer | Strategies | Last loss value |

---

## Creating Custom Strategies

### Step-by-Step Guide

#### Step 1: Choose Strategy Type

Determine which aspect of training you need to customize:

- **Loss computation** → `LossCriterionStrategy`
- **Optimizer setup** → `OptimizerStrategy`
- **Training step** → `TrainingStepStrategy`
- **LR scheduling** → `LRSchedulerStrategy`
- **State management** → `ModelUpdateStrategy`
- **Data loading** → `DataLoaderStrategy`

#### Step 2: Create Strategy Class

```python
from plato.trainers.strategies.base import LossCriterionStrategy, TrainingContext
import torch
import torch.nn as nn

class MyCustomLossStrategy(LossCriterionStrategy):
    """
    Custom loss strategy with L2 regularization.

    This strategy adds L2 regularization to the base loss.

    Args:
        weight: Regularization weight (default: 0.01)
        base_loss_fn: Base loss function (default: CrossEntropyLoss)

    Example:
        >>> strategy = MyCustomLossStrategy(weight=0.01)
        >>> trainer = ComposableTrainer(loss_strategy=strategy)
    """

    def __init__(self, weight=0.01, base_loss_fn=None):
        self.weight = weight
        self.base_loss_fn = base_loss_fn
        self._criterion = None

    def setup(self, context: TrainingContext):
        """Initialize loss criterion."""
        if self.base_loss_fn is None:
            self._criterion = nn.CrossEntropyLoss()
        else:
            self._criterion = self.base_loss_fn

    def compute_loss(self, outputs, labels, context):
        """Compute loss with L2 regularization."""
        # Base loss
        base_loss = self._criterion(outputs, labels)

        # L2 regularization
        l2_reg = 0.0
        for param in context.model.parameters():
            l2_reg += torch.norm(param, p=2)

        return base_loss + self.weight * l2_reg
```

#### Step 3: Use Your Strategy

```python
from plato.trainers.composable import ComposableTrainer

trainer = ComposableTrainer(
    model=my_model,
    loss_strategy=MyCustomLossStrategy(weight=0.01)
)
```

### Advanced Example: State Management Strategy

```python
from plato.trainers.strategies.base import ModelUpdateStrategy, TrainingContext
import copy
import torch

class EMAUpdateStrategy(ModelUpdateStrategy):
    """
    Exponential moving average (EMA) of model weights.

    Maintains an EMA of model weights and optionally uses it for inference.

    Args:
        decay: EMA decay rate (default: 0.999)
        use_ema_for_inference: Use EMA weights for testing (default: True)
    """

    def __init__(self, decay=0.999, use_ema_for_inference=True):
        self.decay = decay
        self.use_ema_for_inference = use_ema_for_inference
        self.ema_weights = None

    def setup(self, context: TrainingContext):
        """Initialize EMA weights."""
        self.ema_weights = copy.deepcopy(context.model.state_dict())

    def on_train_start(self, context: TrainingContext):
        """Save original weights before training."""
        context.state['original_weights'] = copy.deepcopy(
            context.model.state_dict()
        )

    def after_step(self, context: TrainingContext):
        """Update EMA after each step."""
        with torch.no_grad():
            for name, param in context.model.named_parameters():
                if name in self.ema_weights:
                    ema_param = self.ema_weights[name]
                    ema_param.mul_(self.decay).add_(
                        param.data, alpha=1 - self.decay
                    )

    def on_train_end(self, context: TrainingContext):
        """Optionally switch to EMA weights."""
        if self.use_ema_for_inference:
            # Save current weights
            context.state['trained_weights'] = copy.deepcopy(
                context.model.state_dict()
            )
            # Load EMA weights for inference
            context.model.load_state_dict(self.ema_weights)

trainer = ComposableTrainer(
    model=my_model,
    model_update_strategy=EMAUpdateStrategy(decay=0.999)
)
```

### Testing Custom Strategies

```python
import torch
import torch.nn as nn
from plato.trainers.strategies.base import TrainingContext

def test_my_custom_loss_strategy():
    # Create strategy
    strategy = MyCustomLossStrategy(weight=0.01)

    # Create mock context
    context = TrainingContext()
    context.model = nn.Linear(10, 2)
    context.device = torch.device('cpu')

    # Setup strategy
    strategy.setup(context)

    # Test compute_loss
    outputs = torch.randn(8, 2)
    labels = torch.randint(0, 2, (8,))
    loss = strategy.compute_loss(outputs, labels, context)

    # Verify loss is scalar and has gradient
    assert loss.dim() == 0
    assert loss.requires_grad

    # Verify L2 regularization increases loss
    base_loss = nn.CrossEntropyLoss()(outputs, labels)
    assert loss.item() > base_loss.item()

    print("Test passed!")

test_my_custom_loss_strategy()
```

---

## Advanced Usage of Strategy Patterns

### Combining Multiple Strategies

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import (
    FedProxLossStrategy,
    LGFedAvgStepStrategy,
)
from plato.trainers.strategies import (
    AdamOptimizerStrategy,
    CosineAnnealingLRSchedulerStrategy,
)

# Combine FedProx, LG-FedAvg, Adam, and cosine annealing
trainer = ComposableTrainer(
    model=my_model,
    loss_strategy=FedProxLossStrategy(mu=0.01),
    optimizer_strategy=AdamOptimizerStrategy(lr=0.001),
    training_step_strategy=LGFedAvgStepStrategy(
        global_layer_names=['conv', 'fc1'],
        local_layer_names=['fc2']
    ),
    lr_scheduler_strategy=CosineAnnealingLRSchedulerStrategy(T_max=50)
)
```

### Runtime Strategy Swapping

```python
# Create trainer with initial strategies
trainer = ComposableTrainer(
    loss_strategy=FedProxLossStrategy(mu=0.01)
)

# Train for some rounds
trainer.train(trainset, sampler)

# Swap to different strategy
from plato.trainers.strategies.algorithms import FedDynLossStrategy
trainer.loss_strategy = FedDynLossStrategy(alpha=0.01)

# Continue training with new strategy
trainer.train(trainset, sampler)
```

### Config-Based Strategy Selection

```python
from plato.config import Config

def create_trainer_from_config():
    config = Config()

    # Select strategy based on config
    if config.algorithm.name == "FedProx":
        from plato.trainers.strategies.algorithms import (
            FedProxLossStrategyFromConfig
        )
        loss_strategy = FedProxLossStrategyFromConfig()
    elif config.algorithm.name == "FedDyn":
        from plato.trainers.strategies.algorithms import (
            FedDynLossStrategyFromConfig,
            FedDynUpdateStrategy
        )
        loss_strategy = FedDynLossStrategyFromConfig()
        model_update_strategy = FedDynUpdateStrategy()
    else:
        loss_strategy = None
        model_update_strategy = None

    return ComposableTrainer(
        loss_strategy=loss_strategy,
        model_update_strategy=model_update_strategy
    )

trainer = create_trainer_from_config()
```

### Strategy Factories

```python
from plato.trainers.strategies.algorithms import *

class StrategyFactory:
    """Factory for creating common strategy combinations."""

    @staticmethod
    def create_fedprox_trainer(mu=0.01):
        return ComposableTrainer(
            loss_strategy=FedProxLossStrategy(mu=mu)
        )

    @staticmethod
    def create_scaffold_trainer():
        return ComposableTrainer(
            model_update_strategy=SCAFFOLDUpdateStrategy()
        )

    @staticmethod
    def create_feddyn_trainer(alpha=0.01):
        return ComposableTrainer(
            loss_strategy=FedDynLossStrategy(alpha=alpha),
            model_update_strategy=FedDynUpdateStrategy()
        )

    @staticmethod
    def create_lgfedavg_trainer(global_layers, local_layers):
        return ComposableTrainer(
            training_step_strategy=LGFedAvgStepStrategy(
                global_layer_names=global_layers,
                local_layer_names=local_layers
            )
        )

# Use factory
trainer = StrategyFactory.create_fedprox_trainer(mu=0.01)
```

### Composite Loss Strategies

```python
from plato.trainers.strategies.base import LossCriterionStrategy, TrainingContext
import torch.nn as nn

class CompositeLossStrategy(LossCriterionStrategy):
    """Combine multiple loss strategies with weights."""

    def __init__(self, strategies_and_weights):
        """
        Args:
            strategies_and_weights: List of (strategy, weight) tuples
        """
        self.strategies_and_weights = strategies_and_weights

    def setup(self, context: TrainingContext):
        for strategy, _ in self.strategies_and_weights:
            strategy.setup(context)

    def compute_loss(self, outputs, labels, context):
        total_loss = 0.0
        for strategy, weight in self.strategies_and_weights:
            loss = strategy.compute_loss(outputs, labels, context)
            total_loss += weight * loss
        return total_loss

# Use composite loss
from plato.trainers.strategies import CrossEntropyLossStrategy
from plato.trainers.strategies.algorithms import FedProxLossStrategy

composite = CompositeLossStrategy([
    (CrossEntropyLossStrategy(), 1.0),
    (FedProxLossStrategy(mu=0.01), 0.5),
])

trainer = ComposableTrainer(loss_strategy=composite)
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

### Core Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `ComposableTrainer` | `plato.trainers.composable` | Main trainer with strategy support |
| `TrainingContext` | `plato.trainers.strategies.base` | Shared state container |

### Strategy Base Classes

| Strategy | Location | Method to Implement |
|----------|----------|---------------------|
| `LossCriterionStrategy` | `plato.trainers.strategies.base` | `compute_loss()` |
| `OptimizerStrategy` | `plato.trainers.strategies.base` | `create_optimizer()` |
| `TrainingStepStrategy` | `plato.trainers.strategies.base` | `training_step()` |
| `LRSchedulerStrategy` | `plato.trainers.strategies.base` | `create_scheduler()` |
| `ModelUpdateStrategy` | `plato.trainers.strategies.base` | `on_train_start/end()`, `before/after_step()` |
| `DataLoaderStrategy` | `plato.trainers.strategies.base` | `create_train_loader()` |

### Algorithm Strategies

| Algorithm | Strategies | Location |
|-----------|-----------|----------|
| **FedProx** | `FedProxLossStrategy` | `...algorithms.fedprox_strategy` |
| **SCAFFOLD** | `SCAFFOLDUpdateStrategy` | `...algorithms.scaffold_strategy` |
| **FedDyn** | `FedDynLossStrategy`, `FedDynUpdateStrategy` | `...algorithms.feddyn_strategy` |
| **LG-FedAvg** | `LGFedAvgStepStrategy` | `...algorithms.lgfedavg_strategy` |
| **FedMos** | `FedMosOptimizerStrategy`, `FedMosUpdateStrategy` | `...algorithms.fedmos_strategy` |
| **FedPer** | `FedPerUpdateStrategy` | `...algorithms.personalized_fl_strategy` |
| **FedRep** | `FedRepUpdateStrategy` | `...algorithms.personalized_fl_strategy` |
| **APFL** | `APFLUpdateStrategy`, `APFLStepStrategy` | `...algorithms.apfl_strategy` |
| **Ditto** | `DittoUpdateStrategy` | `...algorithms.ditto_strategy` |

---

## Common Patterns

### Access Model in Strategy

```python
def compute_loss(self, outputs, labels, context):
    model = context.model  # Access model
    device = context.device  # Access device
    # Use model and device...
```

### Share Data Between Strategies

```python
# Strategy 1: Store data
def on_train_start(self, context):
    context.state['my_data'] = some_value

# Strategy 2: Read data
def on_train_end(self, context):
    data = context.state.get('my_data')
```

### Combine Multiple Strategies

```python
composite = CompositeLossStrategy([
    (strategy1, weight1),
    (strategy2, weight2),
])
```

---

## Customizing Trainers using Callbacks

For infrastructure changes, such as logging, recording metrics, and stopping the training loop early, we tend to customize the training loop using callbacks instead. The advantage of using callbacks is that one can pass a list of multiple callbacks to the trainer when it is initialized, and they will be called in their order in the provided list. This helps when it is necessary to group features into different callback classes.

Within the implementation of these callback methods, one can access additional information about the training loop by using the `trainer` instance. For example, `trainer.sampler` can be used to access the sampler used by the train dataloader, `trainer.trainloader` can be used to access the current train dataloader, and `trainer.current_epoch` can be used to access the current epoch number.

To use callbacks, subclass the `TrainerCallback` class in `plato.callbacks.trainer`, and override the following methods, then pass it to the trainer when it is initialized, or call `trainer.add_callbacks` after initialization. For built-in trainers that user has no access to the initialization, one can also pass the trainer callbacks to client through parameter `trainer_callbacks`, which will be delivered to trainers later. Examples can be found in `examples/callbacks`.

!!! example "**on_train_run_start()**"
    **`def on_train_run_start(self, trainer, config)`**

    Override this method to complete additional tasks before the training loop starts.

    `trainer` the trainer instance that activated this callback upon the occurrence of the corresponding event.

    `config` the configuration settings used in the training loop. It corresponds directly to the `trainer` section in the configuration file.

    **Example:**

    ```py
    def on_train_run_start(self, trainer, config):
        logging.info(
            "[Client #%d] Loading the dataset with size %d.",
            trainer.client_id,
            len(list(trainer.sampler)),
        )
    ```

!!! example "**on_train_run_end()**"
    **`def on_train_run_end(self, trainer, config)`**

    Override this method to complete additional tasks after the training loop ends.

    `trainer` the trainer instance that activated this callback upon the occurrence of the corresponding event.

    `config` the configuration settings used in the training loop. It corresponds directly to the `trainer` section in the configuration file.

    **Example:**

    ```py
    def on_train_run_end(self, trainer, config):
        logging.info("[Client #%d] Completed the training loop.", trainer.client_id)
    ```

!!! example "**on_train_epoch_start()**"
    **`def on_train_epoch_start(self, trainer, config)`**

    Override this method to complete additional tasks at the starting point of each training epoch.

    `trainer` the trainer instance that activated this callback upon the occurrence of the corresponding event.

    `config` the configuration settings used in the training loop. It corresponds directly to the `trainer` section in the configuration file.

    **Example:**

    ```py
    def train_epoch_start(self, trainer, config):
        logging.info("[Client #%d] Started training epoch %d.", trainer.client_id, trainer.current_epoch)
    ```

!!! example "**on_train_epoch_end()**"
    **`def on_train_epoch_end(self, trainer, config)`**

    Override this method to complete additional tasks at the end of each training epoch.

    `trainer` the trainer instance that activated this callback upon the occurrence of the corresponding event.

    `config` the configuration settings used in the training loop. It corresponds directly to the `trainer` section in the configuration file.

    **Example:**

    ```py
    def on_train_epoch_end(self, trainer, config):
        logging.info("[Client #%d] Finished training epoch %d.", trainer.client_id, trainer.current_epoch)
    ```

!!! example "**on_train_step_start()**"
    **`def on_train_step_start(self, trainer, config, batch=None)`**

    Override this method to complete additional tasks at the beginning of each step within a training epoch.

    `trainer` the trainer instance that activated this callback upon the occurrence of the corresponding event.

    `config` the configuration settings used in the training loop. It corresponds directly to the `trainer` section in the configuration file.

    `batch` the index of the current batch of data that has just been processed in the current step.

    **Example:**

    ```py
    def on_train_step_start(self, trainer, config, batch):
        logging.info("[Client #%d] Started training epoch %d batch %d.", trainer.client_id, trainer.current_epoch, batch)
    ```

!!! example "**on_train_step_end()**"
    **`def on_train_step_end(self, trainer, config, batch=None, loss=None)`**

    Override this method to complete additional tasks at the end of each step within a training epoch.

    `trainer` the trainer instance that activated this callback upon the occurrence of the corresponding event.

    `config` the configuration settings used in the training loop. It corresponds directly to the `trainer` section in the configuration file.

    `batch` the index of the current batch of data that has just been processed in the current step.

    `loss` the loss value computed using the current batch of data after training.

    **Example:**

    ```py
    def on_train_step_end(self, trainer, config, batch, loss):
        logging.info(
            "[Client #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
            trainer.client_id,
            trainer.current_epoch,
            config["epochs"],
            batch,
            len(trainer.train_loader),
            loss.data.item(),
        )
    ```

---

## Accessing and Customizing the Run History During Training

An instance of the `plato.trainers.tracking.RunHistory` class, called `self.run_history`, is used to store any number of performance metrics during the training process, one iterable list of values for each performance metric. By default, it stores the average loss values in each epoch.

The run history in the trainer can be accessed by the client as well, using `self.trainer.run_history`.  It can also be read, updated, or reset in the hooks or callback methods. For example, in the implementation of some algorithms such as Oort, a per-step loss value needs to be stored by calling `update_metric()` in `train_step_end()`:

```py
def train_step_end(self, config, batch=None, loss=None):
    self.run_history.update_metric("train_loss_step", loss.cpu().detach().numpy())
```

Here is a list of all the methods available in the `RunHistory` class:

!!! example "**get_metric_names()**"
    **`def get_metric_names(self)`**

    Returns an iterable set containing of all unique metric names which are being tracked.

!!! example "**get_metric_values()**"
    **`def get_metric_values(self, metric_name)`**

    Returns an ordered iterable list of values that has been stored since the last reset corresponding to the provided metric name.

!!! example "**get_latest_metric()**"
    **`def get_latest_metric(self, metric_name)`**

    Returns the most recent value that has been recorded for the given metric.

!!! example "**update_metric()**"
    **`def update_metric(self, metric_name, metric_value)`**

    Records a new value for the given metric.

!!! example "**reset()**"
    **`def reset(self)`**

    Resets the run history.

---

## Customizing Trainers using Subclassing and Hooks

When using the strategy pattern is no longer feasible, it is also possible to customize the training or testing procedure using subclassing, and overriding hook methods. To customize the training loop using subclassing, subclass the `basic.Trainer` class in `plato.trainers`, and override the following hook methods:

!!! example "train_model()"
    **`def train_model(self, config, trainset, sampler, **kwargs):`**

    Override this method to provide a custom training loop.

    `config` A dictionary of configuration parameters.
    `trainset` The training dataset.
    `sampler` the sampler that extracts a partition for this client.

    **Example:** A complete example can be found in the Hugging Face trainer, located at `plato/trainers/huggingface.py`.

!!! example "test_model()"
    **`test_model(self, config, testset, sampler=None, **kwargs):`**

    Override this method to provide a custom testing loop.

    `config` A dictionary of configuration parameters.
    `testset` The test dataset.

    **Example:** A complete example can be found in `plato/trainers/huggingface.py`.

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

## Frequently Asked Questions

**Q: When should I use strategies vs. callbacks?**
A: Use strategies for algorithmic variations (loss, optimizer, training step). Use callbacks for event-driven behavior (logging, checkpointing).

**Q: Can I use multiple strategies of the same type?**
A: Use `CompositeLossStrategy` or `CompositeUpdateStrategy` to combine multiple strategies.

**Q: Can strategies access the training loop?**
A: Strategies receive a `TrainingContext` with model, device, config, and shared state.
