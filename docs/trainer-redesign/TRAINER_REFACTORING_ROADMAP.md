# Trainer Refactoring Implementation Roadmap

## Executive Summary

This document provides a detailed, actionable roadmap for refactoring the Plato trainer architecture from inheritance-based to composition-based design using the Strategy pattern and Dependency Injection.

**Goal**: Replace inheritance with composition to improve flexibility, testability, and maintainability.

**Timeline**: 14 weeks (3.5 months)

**Risk Level**: Medium (mitigated by backward compatibility)

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Phase-by-Phase Implementation](#phase-by-phase-implementation)
3. [File Structure](#file-structure)
4. [Testing Strategy](#testing-strategy)
5. [Migration Checklist](#migration-checklist)
6. [Best Practices](#best-practices)
7. [FAQ](#faq)

---

## Quick Reference

### Before & After Comparison

#### Before (Inheritance)
```python
class MyTrainer(basic.Trainer):
    def get_loss_criterion(self):
        return custom_loss()

    def train_step_end(self, config, batch=None, loss=None):
        # Custom logic
        super().train_step_end(config, batch, loss)
```

#### After (Composition)
```python
trainer = ComposableTrainer(
    loss_strategy=MyLossStrategy(),
    model_update_strategy=MyUpdateStrategy()
)
```

### Key Benefits
- ✅ **Composability**: Mix multiple strategies
- ✅ **Testability**: Unit test each strategy independently
- ✅ **Flexibility**: Swap strategies at runtime
- ✅ **Reusability**: Share strategies across trainers
- ✅ **Clarity**: Single responsibility per component

---

## Phase-by-Phase Implementation

### Phase 0: Preparation (Week 0)

**Goal**: Set up infrastructure and team alignment

**Tasks**:
1. Review current trainer usage across all examples
2. Identify all extension points and their usage patterns
3. Create tracking spreadsheet for migration progress
4. Set up feature branch: `feature/trainer-refactoring`
5. Configure CI/CD for parallel testing (old + new)

**Deliverables**:
- [ ] Extension point analysis document
- [ ] Team kickoff meeting slides
- [ ] Git branch and CI setup
- [ ] Risk assessment document

**Success Criteria**:
- All team members understand the plan
- Development environment ready
- No blockers identified

---

### Phase 1: Strategy Interface Definition (Week 1-2)

**Goal**: Define all strategy interfaces and default implementations

#### Week 1: Core Interfaces

**Tasks**:
1. Create `plato/trainers/strategies/` directory structure
2. Implement base strategy classes
3. Define `TrainingContext` for shared state
4. Write comprehensive docstrings and type hints

**Files to Create**:
```
plato/trainers/strategies/
├── __init__.py
├── base.py              # Base classes: Strategy, TrainingContext
├── loss_criterion.py    # LossCriterionStrategy + defaults
├── optimizer.py         # OptimizerStrategy + defaults
├── training_step.py     # TrainingStepStrategy + defaults
└── lr_scheduler.py      # LRSchedulerStrategy + defaults
```

**Code Example** (`base.py`):
```python
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
        self.state: Dict[str, Any] = {}

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
```

**Testing**:
- Unit tests for `TrainingContext`
- Abstract method enforcement tests
- Type checking with mypy

**Deliverables**:
- [ ] Base strategy interfaces implemented
- [ ] 100% test coverage on base classes
- [ ] API documentation generated

#### Week 2: Additional Interfaces & Defaults

**Tasks**:
1. Implement `ModelUpdateStrategy` interface
2. Implement `DataLoaderStrategy` interface
3. Create all default implementations
4. Write integration tests

**Files to Create**:
```
plato/trainers/strategies/
├── model_update.py      # ModelUpdateStrategy + NoOpUpdateStrategy
├── data_loader.py       # DataLoaderStrategy + defaults
└── defaults.py          # All default strategy implementations
```

**Testing**:
- Default strategy functionality tests
- Integration tests with mock trainer
- Performance benchmarks (ensure no overhead)

**Deliverables**:
- [ ] All strategy interfaces complete
- [ ] Default implementations tested
- [ ] Performance baseline established

**Success Criteria**:
- All interfaces documented
- No breaking changes to existing code
- Tests passing at 100%

---

### Phase 2: Enhanced Trainer Implementation (Week 3-4)

**Goal**: Create composable trainer that accepts strategies

#### Week 3: Core ComposableTrainer

**Tasks**:
1. Create `plato/trainers/composable.py`
2. Implement strategy injection via constructor
3. Implement training loop with strategy delegation
4. Add backward compatibility detection

**Key Implementation**:
```python
class ComposableTrainer(base.Trainer):
    def __init__(
        self,
        model=None,
        callbacks=None,
        loss_strategy: Optional[LossCriterionStrategy] = None,
        optimizer_strategy: Optional[OptimizerStrategy] = None,
        training_step_strategy: Optional[TrainingStepStrategy] = None,
        lr_scheduler_strategy: Optional[LRSchedulerStrategy] = None,
        model_update_strategy: Optional[ModelUpdateStrategy] = None,
        data_loader_strategy: Optional[DataLoaderStrategy] = None,
    ):
        super().__init__()

        # Initialize context
        self.context = TrainingContext()

        # Initialize strategies with defaults
        self.loss_strategy = loss_strategy or DefaultLossCriterionStrategy()
        self.optimizer_strategy = optimizer_strategy or DefaultOptimizerStrategy()
        # ... etc.

        # Setup all strategies
        for strategy in self._get_all_strategies():
            strategy.setup(self.context)

    def train_model(self, config, trainset, sampler, **kwargs):
        """Training loop using strategies."""
        # Strategy hook: on_train_start
        if self.model_update_strategy:
            self.model_update_strategy.on_train_start(self.context)

        # Create data loader using strategy
        self.train_loader = self.data_loader_strategy.create_train_loader(
            trainset, sampler, batch_size, self.context
        )

        # Training epochs
        for self.current_epoch in range(1, total_epochs + 1):
            for batch_id, (examples, labels) in enumerate(self.train_loader):
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

        # Strategy hook: on_train_end
        if self.model_update_strategy:
            self.model_update_strategy.on_train_end(self.context)
```

**Testing**:
- End-to-end training with default strategies
- Strategy injection tests
- Context sharing tests

**Deliverables**:
- [ ] ComposableTrainer fully functional
- [ ] Compatible with existing datasets
- [ ] Documentation with examples

#### Week 4: Backward Compatibility Layer

**Tasks**:
1. Update `basic.Trainer` to support both approaches
2. Implement legacy method detection
3. Create adapter for old overrides → strategies
4. Add deprecation warnings

**Backward Compatibility Approach**:
```python
class Trainer(base.Trainer):
    def __init__(self, model=None, callbacks=None, **strategy_kwargs):
        # Support both old and new approaches
        if strategy_kwargs:
            # New approach: use strategies
            self._init_with_strategies(**strategy_kwargs)
        else:
            # Old approach: support method overrides
            self._init_legacy_mode()

    def _init_with_strategies(self, **strategy_kwargs):
        """Initialize with strategy injection."""
        self.composable_trainer = ComposableTrainer(**strategy_kwargs)
        # Delegate to composable trainer

    def _init_legacy_mode(self):
        """Initialize in legacy mode with method overrides."""
        # Detect which methods are overridden
        if self._is_method_overridden('get_loss_criterion'):
            # Wrap as strategy
            self.loss_strategy = LegacyMethodAdapter(self.get_loss_criterion)
```

**Testing**:
- All existing examples run without changes
- Deprecation warnings appear correctly
- No performance regression

**Deliverables**:
- [ ] Backward compatibility layer working
- [ ] All existing tests pass
- [ ] Migration guide draft

**Success Criteria**:
- Zero breaking changes to existing code
- New trainer passes all existing tests
- Documentation complete

---

### Phase 3: Algorithm Strategy Implementations (Week 5-8)

**Goal**: Implement strategies for major federated learning algorithms

#### Week 5-6: Core Algorithms

**Priority 1 Algorithms** (implement first):
1. **FedProx** - Simple loss modification
2. **SCAFFOLD** - Complex state management
3. **FedDyn** - Loss + state management
4. **LG-FedAvg** - Training step modification

**Files to Create**:
```
plato/trainers/strategies/implementations/
├── __init__.py
├── fedprox_strategy.py       # FedProxLossStrategy
├── scaffold_strategy.py      # SCAFFOLDUpdateStrategy
├── feddyn_strategy.py        # FedDynLossStrategy, FedDynUpdateStrategy
└── lgfedavg_strategy.py      # LGFedAvgStepStrategy
```

**Template for Each Strategy**:
```python
"""
[Algorithm Name] Strategy

Reference:
[Paper citation and link]

Description:
[Brief description of algorithm]
"""

import torch
from plato.trainers.strategies.base import [Strategy], TrainingContext

class [Algorithm]Strategy([Strategy]):
    """
    [Detailed docstring with:
    - Algorithm description
    - Mathematical formulation
    - Usage example
    - Configuration parameters]
    """

    def __init__(self, **params):
        """Initialize with algorithm-specific parameters."""
        pass

    def setup(self, context: TrainingContext):
        """Setup called once at initialization."""
        pass

    # Implement abstract methods...

    def teardown(self, context: TrainingContext):
        """Cleanup at end of training."""
        pass
```

**Testing for Each Strategy**:
- Unit tests with mock model
- Integration tests with real training
- Correctness validation (reproduce paper results)
- Performance benchmarks

**Deliverables (Week 6)**:
- [ ] FedProx strategy implemented and tested
- [ ] SCAFFOLD strategy implemented and tested
- [ ] FedDyn strategy implemented and tested
- [ ] LG-FedAvg strategy implemented and tested

#### Week 7-8: Additional Algorithms

**Priority 2 Algorithms**:
5. **FedMos** - Optimizer modification
6. **Personalized FL** (FedPer, FedRep, FedBABU)
7. **APFL** - Dual model training
8. **Ditto** - Personalized model training

**Files to Create**:
```
plato/trainers/strategies/implementations/
├── fedmos_strategy.py
├── personalized_fl_strategy.py
├── apfl_strategy.py
└── ditto_strategy.py
```

**Deliverables (Week 8)**:
- [ ] 8+ algorithm strategies implemented
- [ ] All strategies have unit tests
- [ ] Integration tests pass
- [ ] Performance validated

**Success Criteria**:
- All implementations match original algorithm behavior
- Tests achieve ≥90% coverage
- Documentation includes usage examples

---

### Phase 4: Example Migration (Week 9-12)

**Goal**: Migrate examples to new approach, validate correctness

#### Week 9-10: Core Examples

**Migration Process** (per example):
1. Study current implementation
2. Identify which strategies are needed
3. Create new version with strategies
4. Run both versions, compare results
5. Document migration

**Priority Examples**:
- `examples/customized_client_training/fedprox/`
- `examples/customized_client_training/scaffold/`
- `examples/customized_client_training/feddyn/`
- `examples/personalized_fl/lgfedavg/`
- `examples/personalized_fl/fedper/`

**Migration Template**:
```python
# examples/[algorithm]/[algorithm]_v2.py

"""
[Algorithm] implementation using composable trainer with strategies.

This is the refactored version using composition instead of inheritance.
For the legacy version, see [algorithm].py
"""

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import [Strategy]
from plato.clients import simple
from plato.servers import fedavg


def main():
    """Run [Algorithm] with strategy-based trainer."""

    # Create trainer with strategies
    trainer = ComposableTrainer(
        loss_strategy=[Strategy](...),
        # ... other strategies
    )

    # Rest remains the same
    client = simple.Client(trainer=trainer)
    server = fedavg.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
```

**Validation Checklist** (per example):
- [ ] New version runs without errors
- [ ] Results match original implementation (±1% tolerance)
- [ ] Training time comparable (±10% tolerance)
- [ ] Memory usage comparable (±10% tolerance)
- [ ] README updated with both approaches

**Deliverables (Week 10)**:
- [ ] 5 core examples migrated
- [ ] Side-by-side comparison docs
- [ ] Performance benchmarks

#### Week 11-12: Comprehensive Migration

**Tasks**:
1. Migrate remaining high-priority examples (10+)
2. Create migration automation scripts
3. Update all example READMEs
4. Create video tutorials

**Examples to Migrate**:
- All `customized_client_training/` examples
- All `personalized_fl/` examples
- Key `model_search/` examples
- Key `secure_aggregation/` examples

**Automation Script**:
```python
# scripts/migrate_trainer.py

"""
Semi-automated trainer migration tool.

Analyzes an old trainer and suggests strategy-based replacement.
"""

import ast
import argparse

def analyze_trainer(file_path):
    """Analyze trainer to identify overridden methods."""
    with open(file_path) as f:
        tree = ast.parse(f.read())

    # Find Trainer class
    # Identify overridden methods
    # Suggest corresponding strategies
    # Generate migration template

def suggest_strategies(overridden_methods):
    """Suggest strategies based on overridden methods."""
    suggestions = {
        'get_loss_criterion': 'CustomLossCriterionStrategy',
        'get_optimizer': 'CustomOptimizerStrategy',
        'perform_forward_and_backward_passes': 'CustomTrainingStepStrategy',
        'train_step_end': 'CustomModelUpdateStrategy',
        # ... etc.
    }
    return [suggestions.get(m) for m in overridden_methods]

def generate_migration_template(strategies):
    """Generate code template for new implementation."""
    # Generate imports
    # Generate trainer instantiation
    # Generate TODO comments for manual work
```

**Deliverables (Week 12)**:
- [ ] 15+ examples migrated
- [ ] Migration tool script
- [ ] Video tutorials (3-5 videos)
- [ ] Performance comparison report

**Success Criteria**:
- All migrated examples produce equivalent results
- Documentation is comprehensive
- Migration path is clear

---

### Phase 5: Documentation & Finalization (Week 13-14)

**Goal**: Complete documentation, deprecation plan, release

#### Week 13: Documentation

**Tasks**:
1. Write comprehensive developer guide
2. Create tutorial: "Building Custom Trainers"
3. Update API documentation
4. Create migration guide for users
5. Write design rationale document

**Documentation Structure**:
```
docs/trainer_refactoring/
├── overview.md                 # High-level overview
├── design_rationale.md         # Why we did this
├── api_reference.md            # Complete API docs
├── tutorial_basics.md          # Getting started
├── tutorial_custom_strategy.md # Building custom strategies
├── tutorial_composition.md     # Combining strategies
├── migration_guide.md          # How to migrate
├── faq.md                      # Common questions
└── examples/                   # Code examples
    ├── simple_custom_loss.py
    ├── complex_multi_strategy.py
    └── migration_patterns.py
```

**Tutorial: "Building Custom Trainers"**:
```markdown
# Tutorial: Building Custom Trainers with Strategies

## Introduction

This tutorial shows you how to customize trainer behavior using strategies
instead of inheritance.

## Lesson 1: Using Pre-built Strategies

The simplest way to customize training is to use pre-built strategies:

[code example]

## Lesson 2: Creating a Custom Loss Strategy

To implement custom loss logic:

[code example]

## Lesson 3: Combining Multiple Strategies

You can combine strategies to implement complex algorithms:

[code example]

## Lesson 4: Advanced State Management

For algorithms that need to track state (like SCAFFOLD):

[code example]

## Best Practices

- One strategy per concern
- Use context for shared state
- Test strategies independently
- Document strategy parameters
```

**Deliverables**:
- [ ] Developer guide complete
- [ ] Tutorials written and tested
- [ ] API docs generated
- [ ] Migration guide complete

#### Week 14: Deprecation & Release

**Tasks**:
1. Add deprecation warnings to inheritance approach
2. Update CHANGELOG
3. Create release notes
4. Plan deprecation timeline
5. Final testing and QA

**Deprecation Warning Implementation**:
```python
import warnings

class Trainer(base.Trainer):
    def __init__(self, *args, **kwargs):
        # Check if used via inheritance
        if self.__class__ != Trainer:
            if not kwargs.get('_suppress_deprecation_warning'):
                warnings.warn(
                    f"{self.__class__.__name__} extends Trainer via inheritance. "
                    "This approach is deprecated and will be removed in v3.0. "
                    "Please use ComposableTrainer with strategy injection instead. "
                    "See migration guide: https://...",
                    DeprecationWarning,
                    stacklevel=2
                )
```

**Deprecation Timeline**:
- **v1.0 (Current)**: Introduce strategies, mark inheritance as legacy
- **v1.5 (+3 months)**: Deprecation warnings for inheritance
- **v2.0 (+6 months)**: Stronger warnings, migration tools
- **v2.5 (+9 months)**: Final warning, inheritance still works
- **v3.0 (+12 months)**: Remove backward compatibility, strategies only

**Release Checklist**:
- [ ] All tests pass (old + new approaches)
- [ ] Performance validated (no regression)
- [ ] Documentation complete
- [ ] Examples migrated (≥15 examples)
- [ ] Deprecation warnings added
- [ ] CHANGELOG updated
- [ ] Release notes written
- [ ] Blog post drafted
- [ ] Community announcement prepared

**Success Criteria**:
- Zero regression in functionality
- Clear migration path documented
- Community feedback incorporated
- Release artifacts ready

---

## File Structure

### Current Structure
```
plato/
├── trainers/
│   ├── __init__.py
│   ├── base.py              # Abstract base
│   ├── basic.py             # Main implementation (600+ lines)
│   ├── registry.py
│   └── [other specialized trainers]
└── examples/
    └── [40+ examples using inheritance]
```

### New Structure
```
plato/
├── trainers/
│   ├── __init__.py
│   ├── base.py              # Abstract base (unchanged)
│   ├── basic.py             # Legacy + backward compat
│   ├── composable.py        # NEW: Strategy-based trainer
│   ├── registry.py          # Updated
│   ├── strategies/          # NEW: Strategy module
│   │   ├── __init__.py
│   │   ├── base.py          # Strategy interfaces
│   │   ├── loss_criterion.py
│   │   ├── optimizer.py
│   │   ├── training_step.py
│   │   ├── lr_scheduler.py
│   │   ├── model_update.py
│   │   ├── data_loader.py
│   │   ├── defaults.py      # Default implementations
│   │   ├── factories.py     # Factory methods
│   │   ├── builders.py      # Builder pattern
│   │   └── implementations/ # Algorithm strategies
│   │       ├── __init__.py
│   │       ├── fedprox_strategy.py
│   │       ├── scaffold_strategy.py
│   │       ├── feddyn_strategy.py
│   │       ├── lgfedavg_strategy.py
│   │       ├── fedmos_strategy.py
│   │       ├── personalized_fl_strategy.py
│   │       └── [15+ more strategies]
│   └── [other specialized trainers]
├── examples/
│   ├── [original examples] # Keep for backward compat
│   └── [algorithm]/
│       ├── [algorithm].py     # Original (legacy)
│       └── [algorithm]_v2.py  # NEW: Strategy-based
└── docs/
    └── trainer_refactoring/   # NEW: Documentation
        ├── overview.md
        ├── api_reference.md
        ├── migration_guide.md
        └── tutorials/
```

---

## Testing Strategy

### Test Pyramid

```
            [E2E Tests]           <- 10% (Full training runs)
          /            \
         /  Integration  \        <- 30% (Strategy + Trainer)
        /                 \
       /   Unit Tests      \      <- 60% (Individual strategies)
      /____________________\
```

### Unit Tests (60% of tests)

**Scope**: Individual strategies in isolation

**Example**:
```python
# tests/trainers/strategies/test_fedprox_strategy.py

import torch
import torch.nn as nn
from plato.trainers.strategies.algorithms import FedProxLossStrategy
from plato.trainers.strategies.base import TrainingContext

class TestFedProxLossStrategy:

    def test_initialization(self):
        """Test strategy initializes correctly."""
        strategy = FedProxLossStrategy(mu=0.01)
        assert strategy.mu == 0.01

    def test_compute_loss_without_proximal(self):
        """Test loss equals base loss when mu=0."""
        strategy = FedProxLossStrategy(mu=0.0)
        context = TrainingContext()

        model = nn.Linear(10, 2)
        context.model = model

        strategy.setup(context)

        outputs = torch.randn(32, 2)
        labels = torch.randint(0, 2, (32,))

        loss = strategy.compute_loss(outputs, labels, context)

        # Should equal CrossEntropyLoss when mu=0
        expected = nn.CrossEntropyLoss()(outputs, labels)
        assert torch.allclose(loss, expected)

    def test_proximal_term_increases_loss(self):
        """Test proximal term adds to base loss."""
        strategy = FedProxLossStrategy(mu=0.1)
        context = TrainingContext()

        model = nn.Linear(10, 2)
        context.model = model

        strategy.setup(context)

        outputs = torch.randn(32, 2)
        labels = torch.randint(0, 2, (32,))

        # First call: saves global weights
        loss1 = strategy.compute_loss(outputs, labels, context)

        # Modify model weights
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        # Second call: should have proximal term
        outputs2 = model(torch.randn(32, 10))
        loss2 = strategy.compute_loss(outputs2, labels, context)

        # Loss2 should be different due to proximal term
        assert not torch.allclose(loss1, loss2)

    def test_setup_teardown(self):
        """Test setup and teardown lifecycle."""
        strategy = FedProxLossStrategy(mu=0.01)
        context = TrainingContext()
        context.model = nn.Linear(10, 2)

        assert strategy.global_weights is None

        strategy.setup(context)
        # setup should be idempotent

        strategy.teardown(context)
        assert strategy.global_weights is None
```

**Coverage Target**: ≥95% line coverage per strategy

### Integration Tests (30% of tests)

**Scope**: Strategy + Trainer working together

**Example**:
```python
# tests/trainers/test_composable_trainer.py

import torch
from torch.utils.data import TensorDataset
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import FedProxLossStrategy

class TestComposableTrainer:

    def test_training_with_fedprox(self):
        """Test end-to-end training with FedProx strategy."""

        # Create simple dataset
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        dataset = TensorDataset(X, y)

        # Create trainer with FedProx
        trainer = ComposableTrainer(
            model=lambda: torch.nn.Linear(10, 2),
            loss_strategy=FedProxLossStrategy(mu=0.01)
        )

        # Configure
        config = {
            'batch_size': 32,
            'epochs': 2,
            'lr': 0.01,
        }

        # Train
        sampler = list(range(len(dataset)))
        trainer.train_model(config, dataset, sampler)

        # Verify training occurred
        assert trainer.current_epoch == 2
        assert len(trainer.run_history.get_metric('train_loss')) > 0

    def test_strategy_receives_context(self):
        """Test that strategies receive correct context."""

        class MockStrategy(LossCriterionStrategy):
            def __init__(self):
                self.context_received = None

            def compute_loss(self, outputs, labels, context):
                self.context_received = context
                return torch.tensor(0.0)

        mock_strategy = MockStrategy()
        trainer = ComposableTrainer(loss_strategy=mock_strategy)

        # ... train ...

        assert mock_strategy.context_received is not None
        assert mock_strategy.context_received.model is trainer.model
```

**Coverage Target**: ≥85% line coverage

### End-to-End Tests (10% of tests)

**Scope**: Full training runs with real models and datasets

**Example**:
```python
# tests/integration/test_fedprox_full.py

import pytest
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import FedProxLossStrategy
from plato.datasources import registry as datasource_registry
from plato.config import Config

@pytest.mark.slow
class TestFedProxFullTraining:

    def test_fedprox_mnist(self):
        """Test FedProx on MNIST achieves expected accuracy."""

        # Configure
        Config().trainer.epochs = 5
        Config().trainer.batch_size = 32

        # Create datasource
        datasource = datasource_registry.get(client_id=1)
        trainset = datasource.get_train_set()

        # Create trainer
        trainer = ComposableTrainer(
            loss_strategy=FedProxLossStrategy(mu=0.01)
        )

        # Train
        trainer.train(trainset, sampler=None)

        # Test
        testset = datasource.get_test_set()
        accuracy = trainer.test(testset)

        # Verify reasonable accuracy
        assert accuracy > 0.85  # Should achieve >85% on MNIST
```

**Coverage Target**: Major algorithms tested on real datasets

### Continuous Testing

**CI Pipeline**:
```yaml
# .github/workflows/test.yml

name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest tests/trainers/strategies/ -v --cov

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run integration tests
        run: pytest tests/trainers/ -v --cov

  backward-compat-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Test old examples still work
        run: |
          pytest tests/examples/ -v
          python examples/customized_client_training/fedprox/fedprox.py

  performance-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run performance benchmarks
        run: python tests/benchmarks/trainer_performance.py
```

---

## Migration Checklist

### For Framework Developers

- [ ] **Phase 1**: Strategy interfaces defined
  - [ ] All interfaces have docstrings
  - [ ] Type hints complete
  - [ ] Default implementations working
  - [ ] Unit tests at 95%+ coverage

- [ ] **Phase 2**: Composable trainer implemented
  - [ ] Strategy injection working
  - [ ] Training loop delegates to strategies
  - [ ] Backward compatibility layer working
  - [ ] All existing tests pass

- [ ] **Phase 3**: Algorithm strategies implemented
  - [ ] ≥8 major algorithms converted
  - [ ] Each strategy tested independently
  - [ ] Performance validated
  - [ ] Documentation complete

- [ ] **Phase 4**: Examples migrated
  - [ ] ≥15 examples migrated
  - [ ] Side-by-side comparisons
  - [ ] Results validated
  - [ ] Migration tool created

- [ ] **Phase 5**: Documentation & release
  - [ ] API documentation complete
  - [ ] Tutorials written
  - [ ] Migration guide complete
  - [ ] Deprecation warnings added
  - [ ] Release notes written

### For Algorithm Developers

When migrating an algorithm from inheritance to strategies:

- [ ] **Analyze** current implementation
  - [ ] Identify
