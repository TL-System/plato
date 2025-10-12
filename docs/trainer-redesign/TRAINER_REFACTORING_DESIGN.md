# Trainer Refactoring Design: From Inheritance to Composition

## Executive Summary

This document proposes a comprehensive refactoring of the Plato federated learning framework's trainer architecture, replacing the current inheritance-based extension mechanism with a composition-based approach using the Strategy pattern and Dependency Injection (DI). This change will improve flexibility, testability, and maintainability while maintaining backward compatibility.

## Table of Contents

1. [Current Architecture Analysis](#current-architecture-analysis)
2. [Problems with Current Approach](#problems-with-current-approach)
3. [Proposed Architecture](#proposed-architecture)
4. [Design Patterns](#design-patterns)
5. [Implementation Plan](#implementation-plan)
6. [Migration Strategy](#migration-strategy)
7. [Code Examples](#code-examples)
8. [Testing Strategy](#testing-strategy)
9. [Backward Compatibility](#backward-compatibility)

---

## Current Architecture Analysis

### Inheritance Hierarchy

```
base.Trainer (Abstract)
    ↓
basic.Trainer (Concrete - ~600 lines)
    ↓
[40+ Custom Trainers in examples/]
    - FedProx, SCAFFOLD, FedDyn, etc.
```

### Extension Points (Currently via Method Override)

1. **Loss Computation**: `get_loss_criterion()`, `process_loss()`
2. **Optimizer Configuration**: `get_optimizer()`
3. **Training Step Logic**: `perform_forward_and_backward_passes()`
4. **Training Loop**: `train_model()` (complete override)
5. **Lifecycle Hooks**: `train_run_start()`, `train_epoch_start()`, `train_step_end()`, etc.
6. **LR Scheduling**: `get_lr_scheduler()`, `lr_scheduler_step()`

### Current Extension Mechanisms

1. **Inheritance**: Subclass `basic.Trainer` and override methods
2. **Callbacks**: Event-driven hooks for lifecycle events (good pattern, keep it)

---

## Problems with Current Approach

### 1. Tight Coupling
- Subclasses depend on internal implementation details of base class
- Changes to `basic.Trainer` can break custom trainers
- Difficult to understand what a subclass actually changes

### 2. Limited Composability
- Cannot combine multiple algorithms (e.g., FedProx loss + SCAFFOLD updates)
- Each combination requires a new subclass
- Code duplication across similar trainers

### 3. Testing Challenges
- Cannot test individual components in isolation
- Must instantiate full trainer to test single feature
- Mocking is difficult due to tight coupling

### 4. Fragile Base Class Problem
- Base class changes affect all subclasses
- Cannot safely refactor base implementation
- Fear of breaking existing code prevents improvements

### 5. Runtime Inflexibility
- Cannot change behavior at runtime
- Strategy must be chosen at class definition time
- No dynamic algorithm selection

### 6. Violation of SOLID Principles
- **Single Responsibility**: Trainer handles too many concerns
- **Open/Closed**: Must modify base class for new features
- **Liskov Substitution**: Subclasses may violate base contracts
- **Interface Segregation**: Large interface with many methods
- **Dependency Inversion**: Depends on concrete implementations

---

## Proposed Architecture

### Core Principle: **Composition over Inheritance**

Instead of overriding methods, inject **strategy objects** that encapsulate specific behaviors.

### Strategy Interfaces

```
TrainerStrategies:
    ├── LossCriterionStrategy      # How to compute loss
    ├── OptimizerStrategy          # How to create/configure optimizer
    ├── TrainingStepStrategy       # How to perform forward/backward pass
    ├── LRSchedulerStrategy        # How to schedule learning rate
    ├── ModelUpdateStrategy        # How to process model updates (SCAFFOLD, etc.)
    └── DataLoaderStrategy         # How to create data loaders
```

### Enhanced Trainer Architecture

```python
class Trainer:
    def __init__(
        self,
        model=None,
        callbacks=None,
        # Strategy injection
        loss_strategy: Optional[LossCriterionStrategy] = None,
        optimizer_strategy: Optional[OptimizerStrategy] = None,
        training_step_strategy: Optional[TrainingStepStrategy] = None,
        lr_scheduler_strategy: Optional[LRSchedulerStrategy] = None,
        model_update_strategy: Optional[ModelUpdateStrategy] = None,
        data_loader_strategy: Optional[DataLoaderStrategy] = None,
    ):
        # Initialize with defaults or injected strategies
        self.loss_strategy = loss_strategy or DefaultLossCriterionStrategy()
        self.optimizer_strategy = optimizer_strategy or DefaultOptimizerStrategy()
        # ... etc.
```

### Benefits

✅ **Composability**: Combine multiple strategies  
✅ **Testability**: Test strategies independently  
✅ **Flexibility**: Swap strategies at runtime  
✅ **Clarity**: Each strategy has single responsibility  
✅ **Extensibility**: Add new strategies without modifying trainer  
✅ **Reusability**: Strategies work with any trainer  

---

## Design Patterns

### 1. Strategy Pattern

**Intent**: Define a family of algorithms, encapsulate each one, and make them interchangeable.

**Application**: Each extension point becomes a strategy interface.

### 2. Dependency Injection

**Intent**: Provide dependencies from outside rather than creating them internally.

**Application**: Inject strategies via constructor or setters.

### 3. Template Method (Limited Use)

**Intent**: Define skeleton of algorithm, defer some steps to subclasses.

**Application**: Keep for core training loop structure, use strategies for variations.

### 4. Factory Pattern

**Intent**: Create objects without specifying exact class.

**Application**: Strategy factories for common configurations.

### 5. Composite Pattern

**Intent**: Compose objects into tree structures.

**Application**: Chain multiple strategies (e.g., loss decorators).

---

## Implementation Plan

### Phase 1: Strategy Interface Definition (Week 1-2)

**Deliverables**:
- Create `plato/trainers/strategies/` module
- Define all strategy interfaces
- Implement default strategies (current behavior)
- Unit tests for default strategies

**Files to Create**:
```
plato/trainers/strategies/
    ├── __init__.py
    ├── base.py                    # Base strategy classes
    ├── loss_criterion.py          # LossCriterionStrategy
    ├── optimizer.py               # OptimizerStrategy
    ├── training_step.py           # TrainingStepStrategy
    ├── lr_scheduler.py            # LRSchedulerStrategy
    ├── model_update.py            # ModelUpdateStrategy
    ├── data_loader.py             # DataLoaderStrategy
    └── factories.py               # Strategy factory methods
```

### Phase 2: Enhanced Trainer Implementation (Week 3-4)

**Deliverables**:
- Modify `basic.Trainer` to accept strategies
- Implement backward compatibility layer
- Integration tests
- Documentation

**Changes**:
- Update `plato/trainers/basic.py`
- Add `plato/trainers/composable.py` (new strategy-based trainer)
- Update registry to support both approaches

### Phase 3: Strategy Implementations (Week 5-8)

**Deliverables**:
- Implement strategies for all major algorithms
- Create example configurations
- Migration guide

**Strategy Implementations**:
```
plato/trainers/strategies/implementations/
    ├── fedprox_loss_strategy.py
    ├── scaffold_update_strategy.py
    ├── feddyn_step_strategy.py
    ├── fedmos_optimizer_strategy.py
    └── ... (one per algorithm)
```

### Phase 4: Example Migration (Week 9-12)

**Deliverables**:
- Convert 5-10 key examples to new approach
- Side-by-side comparisons
- Performance benchmarks

**Priority Examples**:
1. FedProx (simple loss modification)
2. SCAFFOLD (complex state management)
3. LG-FedAvg (step modification)
4. FedDyn (loss and state)
5. Personalized FL examples

### Phase 5: Documentation & Deprecation (Week 13-14)

**Deliverables**:
- Complete developer documentation
- Tutorial: "Building Custom Trainers"
- Deprecation warnings for old approach
- Migration automation scripts

---

## Migration Strategy

### Backward Compatibility Approach

**Goal**: Existing code continues to work without changes.

**Technique**: Adapter pattern to convert old overrides to strategies.

```python
class Trainer:
    def __init__(self, ...):
        # Check if subclass overrides methods (legacy mode)
        if self._is_method_overridden('get_loss_criterion'):
            # Wrap override as strategy
            self.loss_strategy = LegacyMethodStrategy(self.get_loss_criterion)
        else:
            # Use injected strategy
            self.loss_strategy = loss_strategy or DefaultLossCriterionStrategy()
```

### Migration Levels

#### Level 0: No Changes (Legacy)
```python
# Still works - backward compatible
class MyTrainer(basic.Trainer):
    def get_loss_criterion(self):
        return my_custom_loss
```

#### Level 1: Strategy Injection (Recommended)
```python
# New approach - composition
trainer = basic.Trainer(
    loss_strategy=FedProxLossStrategy(mu=0.01),
    model_update_strategy=SCAFFOLDUpdateStrategy()
)
```

#### Level 2: Custom Strategies (Advanced)
```python
# Create reusable strategy
class MyLossStrategy(LossCriterionStrategy):
    def compute_loss(self, model, outputs, labels, context):
        # Custom loss logic
        return loss

trainer = basic.Trainer(loss_strategy=MyLossStrategy())
```

### Deprecation Timeline

- **v1.0**: Introduce strategies alongside inheritance (6 months)
- **v2.0**: Mark inheritance as deprecated, warnings (6 months)
- **v3.0**: Remove backward compatibility, strategies only (future)

---

## Code Examples

### Strategy Interface Definitions

```python
# plato/trainers/strategies/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
import torch.nn as nn


class TrainingContext:
    """Shared context passed between strategies."""
    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.device: Optional[torch.device] = None
        self.client_id: int = 0
        self.current_epoch: int = 0
        self.current_round: int = 0
        self.config: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}  # For strategies to share state


class Strategy(ABC):
    """Base class for all strategies."""
    
    def setup(self, context: TrainingContext) -> None:
        """Called once during trainer initialization."""
        pass
    
    def teardown(self, context: TrainingContext) -> None:
        """Called when training is complete."""
        pass


class LossCriterionStrategy(Strategy):
    """Strategy for computing loss."""
    
    @abstractmethod
    def compute_loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        context: TrainingContext
    ) -> torch.Tensor:
        """Compute loss given model outputs and labels."""
        pass


class OptimizerStrategy(Strategy):
    """Strategy for creating and configuring optimizer."""
    
    @abstractmethod
    def create_optimizer(
        self,
        model: nn.Module,
        context: TrainingContext
    ) -> torch.optim.Optimizer:
        """Create optimizer for the model."""
        pass
    
    def on_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        context: TrainingContext
    ) -> None:
        """Hook called after optimizer.step()."""
        pass


class TrainingStepStrategy(Strategy):
    """Strategy for performing forward and backward passes."""
    
    @abstractmethod
    def training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        examples: torch.Tensor,
        labels: torch.Tensor,
        loss_criterion: callable,
        context: TrainingContext
    ) -> torch.Tensor:
        """Perform one training step (forward + backward + optimize)."""
        pass


class ModelUpdateStrategy(Strategy):
    """Strategy for processing model updates (e.g., SCAFFOLD control variates)."""
    
    def on_train_start(self, context: TrainingContext) -> None:
        """Called at the start of training."""
        pass
    
    def on_train_end(self, context: TrainingContext) -> None:
        """Called at the end of training."""
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


class LRSchedulerStrategy(Strategy):
    """Strategy for learning rate scheduling."""
    
    @abstractmethod
    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        context: TrainingContext
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        pass
    
    def step(
        self,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        context: TrainingContext
    ) -> None:
        """Perform scheduler step."""
        if scheduler is not None:
            scheduler.step()


class DataLoaderStrategy(Strategy):
    """Strategy for creating data loaders."""
    
    @abstractmethod
    def create_train_loader(
        self,
        trainset,
        sampler,
        batch_size: int,
        context: TrainingContext
    ) -> torch.utils.data.DataLoader:
        """Create training data loader."""
        pass
```

### Default Strategy Implementations

```python
# plato/trainers/strategies/loss_criterion.py

import torch
import torch.nn as nn
from plato.trainers import loss_criterion
from .base import LossCriterionStrategy, TrainingContext


class DefaultLossCriterionStrategy(LossCriterionStrategy):
    """Default loss criterion strategy using framework's registry."""
    
    def __init__(self, loss_fn=None):
        self.loss_fn = loss_fn
        self._criterion = None
    
    def setup(self, context: TrainingContext):
        if self.loss_fn is None:
            self._criterion = loss_criterion.get()
        else:
            self._criterion = self.loss_fn
    
    def compute_loss(self, outputs, labels, context):
        return self._criterion(outputs, labels)


class CrossEntropyLossStrategy(LossCriterionStrategy):
    """Simple cross-entropy loss."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._criterion = None
    
    def setup(self, context: TrainingContext):
        self._criterion = nn.CrossEntropyLoss(**self.kwargs)
    
    def compute_loss(self, outputs, labels, context):
        return self._criterion(outputs, labels)
```

```python
# plato/trainers/strategies/training_step.py

import torch
from .base import TrainingStepStrategy, TrainingContext


class DefaultTrainingStepStrategy(TrainingStepStrategy):
    """Standard training step: forward -> loss -> backward -> step."""
    
    def __init__(self, create_graph=False):
        self.create_graph = create_graph
    
    def training_step(self, model, optimizer, examples, labels, 
                     loss_criterion, context):
        optimizer.zero_grad()
        
        outputs = model(examples)
        loss = loss_criterion(outputs, labels)
        
        loss.backward(create_graph=self.create_graph)
        optimizer.step()
        
        return loss


class AccumulatedGradientStepStrategy(TrainingStepStrategy):
    """Training step with gradient accumulation."""
    
    def __init__(self, accumulation_steps=4):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def training_step(self, model, optimizer, examples, labels,
                     loss_criterion, context):
        outputs = model(examples)
        loss = loss_criterion(outputs, labels)
        
        # Scale loss by accumulation steps
        loss = loss / self.accumulation_steps
        loss.backward()
        
        self.current_step += 1
        if self.current_step % self.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        return loss * self.accumulation_steps  # Return unscaled loss
```

### Algorithm-Specific Strategies

```python
# plato/trainers/strategies/implementations/fedprox_loss_strategy.py

import torch
from plato.trainers.strategies.base import LossCriterionStrategy, TrainingContext


class FedProxLossStrategy(LossCriterionStrategy):
    """
    FedProx loss with proximal term.
    
    Reference:
    Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020.
    """
    
    def __init__(self, mu=0.01, base_loss_fn=None):
        self.mu = mu
        self.base_loss_fn = base_loss_fn or torch.nn.CrossEntropyLoss()
        self.global_weights = None
    
    def setup(self, context: TrainingContext):
        # Save global model weights at start of training
        self.global_weights = {
            name: param.clone().detach()
            for name, param in context.model.named_parameters()
        }
    
    def compute_loss(self, outputs, labels, context):
        # Standard loss
        base_loss = self.base_loss_fn(outputs, labels)
        
        # Proximal term: (mu/2) * ||w - w_global||^2
        proximal_term = 0.0
        for name, param in context.model.named_parameters():
            if name in self.global_weights:
                proximal_term += torch.norm(
                    param - self.global_weights[name].to(param.device)
                ) ** 2
        
        proximal_term = (self.mu / 2) * proximal_term
        
        return base_loss + proximal_term
```

```python
# plato/trainers/strategies/implementations/scaffold_update_strategy.py

import torch
import copy
import logging
import pickle
from collections import OrderedDict
from plato.trainers.strategies.base import ModelUpdateStrategy, TrainingContext
from plato.config import Config


class SCAFFOLDUpdateStrategy(ModelUpdateStrategy):
    """
    SCAFFOLD control variate strategy.
    
    Reference:
    Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging 
    for Federated Learning", ICML 2020.
    """
    
    def __init__(self):
        self.server_control_variate = None
        self.client_control_variate = None
        self.global_model_weights = None
        self.local_steps = 0
        self.learning_rate = None
        self.client_control_variate_path = None
    
    def setup(self, context: TrainingContext):
        model_path = Config().params["model_path"]
        self.client_control_variate_path = (
            f"{model_path}_scaffold_cv_{context.client_id}.pkl"
        )
    
    def on_train_start(self, context: TrainingContext):
        # Receive server control variate from context
        self.server_control_variate = context.state.get('server_control_variate')
        
        # Initialize client control variate if first time
        if self.client_control_variate is None:
            self.client_control_variate = {
                name: torch.zeros_like(param)
                for name, param in context.model.named_parameters()
            }
        
        # Save global model weights
        self.global_model_weights = copy.deepcopy(context.model.state_dict())
        self.local_steps = 0
    
    def after_step(self, context: TrainingContext):
        # Apply control variate correction after optimizer step
        with torch.no_grad():
            for name, param in context.model.named_parameters():
                if name in self.server_control_variate:
                    correction = (
                        self.server_control_variate[name].to(param.device)
                        - self.client_control_variate[name].to(param.device)
                    )
                    param.data.add_(correction, alpha=self.learning_rate)
        
        self.local_steps += 1
    
    def on_train_end(self, context: TrainingContext):
        # Compute new client control variate
        eta = self.learning_rate or Config().trainer.lr
        tau = max(1, self.local_steps)
        
        new_client_cv = OrderedDict()
        delta_cv = OrderedDict()
        
        for name, param in context.model.named_parameters():
            x_global = self.global_model_weights[name]
            x_local = param.data
            c_old = self.client_control_variate[name]
            
            # c_new = c_server - (x_local - x_global) / (eta * tau)
            c_new = (
                self.server_control_variate[name].to(param.device)
                - (x_local - x_global.to(param.device)) / (eta * tau)
            )
            
            new_client_cv[name] = c_new.detach().cpu()
            delta_cv[name] = (c_new - c_old.to(param.device)).detach().cpu()
        
        self.client_control_variate = new_client_cv
        
        # Save client control variate
        with open(self.client_control_variate_path, 'wb') as f:
            pickle.dump(self.client_control_variate, f)
        
        # Store delta for sending to server
        context.state['client_control_variate_delta'] = delta_cv
    
    def get_update_payload(self, context: TrainingContext):
        return {
            'control_variate_delta': context.state.get('client_control_variate_delta')
        }
```

```python
# plato/trainers/strategies/implementations/lgfedavg_step_strategy.py

import torch
from plato.trainers.strategies.base import TrainingStepStrategy, TrainingContext
from plato.config import Config


class LGFedAvgStepStrategy(TrainingStepStrategy):
    """
    LG-FedAvg training step with dual forward/backward passes.
    
    Trains local layers first, then global layers.
    """
    
    def __init__(self, global_layer_names, local_layer_names):
        self.global_layer_names = global_layer_names
        self.local_layer_names = local_layer_names
    
    def _set_requires_grad(self, model, layer_names, requires_grad):
        """Enable/disable gradients for specific layers."""
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = requires_grad
    
    def training_step(self, model, optimizer, examples, labels,
                     loss_criterion, context):
        # First pass: Train local layers only
        self._set_requires_grad(model, self.global_layer_names, False)
        self._set_requires_grad(model, self.local_layer_names, True)
        
        optimizer.zero_grad()
        outputs = model(examples)
        loss = loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Second pass: Train global layers only
        self._set_requires_grad(model, self.global_layer_names, True)
        self._set_requires_grad(model, self.local_layer_names, False)
        
        optimizer.zero_grad()
        outputs = model(examples)
        loss = loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Re-enable all gradients
        self._set_requires_grad(model, self.global_layer_names, True)
        self._set_requires_grad(model, self.local_layer_names, True)
        
        return loss
```

### Enhanced Trainer with Strategy Support

```python
# plato/trainers/composable.py

"""
A composable trainer that uses strategies instead of inheritance.
"""

import time
import torch
from typing import Optional

from plato.trainers import base
from plato.trainers.strategies.base import (
    TrainingContext,
    LossCriterionStrategy,
    OptimizerStrategy,
    TrainingStepStrategy,
    LRSchedulerStrategy,
    ModelUpdateStrategy,
    DataLoaderStrategy,
)
from plato.trainers.strategies.loss_criterion import DefaultLossCriterionStrategy
from plato.trainers.strategies.optimizer import DefaultOptimizerStrategy
from plato.trainers.strategies.training_step import DefaultTrainingStepStrategy
from plato.trainers.strategies.lr_scheduler import DefaultLRSchedulerStrategy
from plato.trainers.strategies.data_loader import DefaultDataLoaderStrategy
from plato.callbacks.handler import CallbackHandler
from plato.callbacks.trainer import LogProgressCallback
from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers import tracking


class ComposableTrainer(base.Trainer):
    """
    A trainer that uses composition and strategy pattern for extensibility.
    
    Instead of overriding methods, inject strategy objects to customize behavior.
    """
    
    def __init__(
        self,
        model=None,
        callbacks=None,
        # Strategy injection
        loss_strategy: Optional[LossCriterionStrategy] = None,
        optimizer_strategy: Optional[OptimizerStrategy] = None,
        training_step_strategy: Optional[TrainingStepStrategy] = None,
        lr_scheduler_strategy: Optional[LRSchedulerStrategy] = None,
        model_update_strategy: Optional[ModelUpdateStrategy] = None,
        data_loader_strategy: Optional[DataLoaderStrategy] = None,
    ):
        """
        Initialize trainer with strategies.
        
        Args:
            model: Model class or instance
            callbacks: List of callback classes or instances
            loss_strategy: Strategy for computing loss
            optimizer_strategy: Strategy for creating optimizer
            training_step_strategy: Strategy for training step
            lr_scheduler_strategy: Strategy for LR scheduling
            model_update_strategy: Strategy for model updates (e.g., SCAFFOLD)
            data_loader_strategy: Strategy for creating data loaders
        """
        super().__init__()
        
        # Initialize context
        self.context = TrainingContext()
        self.context.device = self.device
        self.context.client_id = self.client_id
        
        # Initialize model
        if model is None:
            self.model = models_registry.get()
        else:
            self.model = model() if callable(model) else model
        self.context.model = self.model
        
        # Initialize strategies with defaults
        self.loss_strategy = loss_strategy or DefaultLossCriterionStrategy()
        self.optimizer_strategy = optimizer_strategy or DefaultOptimizerStrategy()
        self.training_step_strategy = (
            training_step_strategy or DefaultTrainingStepStrategy()
        )
        self.lr_scheduler_strategy = (
            lr_scheduler_strategy or DefaultLRSchedulerStrategy()
        )
        self.model_update_strategy = model_update_strategy
        self.data_loader_strategy = (
            data_loader_strategy or DefaultDataLoaderStrategy()
        )
        
        # Setup all strategies
        for strategy in self._get_all_strategies():
            strategy.setup(self.context)
        
        # Initialize callbacks
        self.callbacks = [LogProgressCallback]
        if callbacks is not None:
            self.callbacks.extend(callbacks)
        self.callback_handler = CallbackHandler(self.callbacks)
        
        # Initialize tracking
        self.run_history = tracking.RunHistory()
        self._loss_tracker = tracking.LossTracker()
        
        # Training state
        self.train_loader = None
        self.sampler = None
        self.optimizer = None
        self.lr_scheduler = None
        self.current_epoch = 0
        self.current_round = 0
        self.training_start_time = time.time()
    
    def _get_all_strategies(self):
        """Get all non-None strategies."""
        strategies = [
            self.loss_strategy,
            self.optimizer_strategy,
            self.training_step_strategy,
            self.lr_scheduler_strategy,
            self.data_loader_strategy,
        ]
        if self.model_update_strategy is not None:
            strategies.append(self.model_update_strategy)
        return strategies
    
    def set_client_id(self, client_id):
        """Set client ID."""
        super().set_client_id(client_id)
        self.context.client_id = client_id
    
    def train_model(self, config, trainset, sampler, **kwargs):
        """Main training loop using strategies."""
        batch_size = config["batch_size"]
        self.sampler = sampler
        self.context.config = config
        
        self.run_history.reset()
        self._loss_tracker.reset()
        
        # Callbacks: train run start
        self.callback_handler.call_event("on_train_run_start", self, config)
        
        # Strategy hook: on_train_start
        if self.model_update_strategy:
            self.model_update_strategy.on_train_start(self.context)
        
        # Create data loader using strategy
        self.train_loader = self.data_loader_strategy.create_train_loader(
            trainset, sampler, batch_size, self.context
        )
        
        # Create optimizer using strategy
        self.optimizer = self.optimizer_strategy.create_optimizer(
            self.model, self.context
        )
        
        # Create LR scheduler using strategy
        self.lr_scheduler = self.lr_scheduler_strategy.create_scheduler(
            self.optimizer, self.context
        )
        
        # Move model to device
        self.model.to(self.device)
        self.model.train()
        
        # Training epochs
        total_epochs = config["epochs"]
        for self.current_epoch in range(1, total_epochs + 1):
            self.context.current_epoch = self.current_epoch
            self._loss_tracker.reset()
            
            # Callbacks: epoch start
            self.callback_handler.call_event("on_train_epoch_start", self, config)
            
            # Training steps
            for batch_id, (examples, labels) in enumerate(self.train_loader):
                # Callbacks: step start
                self.callback_handler.call_event(
                    "on_train_step_start", self, config, batch=batch_id
                )
                
                # Strategy hook: before_step
                if self.model_update_strategy:
                    self.model_update_strategy.before_step(self.context)
                
                # Move data to device
                examples = examples.to(self.device)
                labels = labels.to(self.device)
                
                # Perform training step using strategy
                loss = self.training_step_strategy.training_step(
                    model=self.model,
                    optimizer=self.optimizer,
                    examples=examples,
                    labels=labels,
                    loss_criterion=lambda o, l: self.loss_strategy.compute_loss(
                        o, l, self.context
                    ),
                    context=self.context,
                )
                
                # Track loss
                self._loss_tracker.update(loss, labels.size(0))
                
                # Strategy hook: after_step
                if self.model_update_strategy:
                    self.model_update_strategy.after_step(self.context)
                
                # Callbacks: step end
                self.callback_handler.call_event(
                    "on_train