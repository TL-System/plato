# Migration Guide: From Inheritance to Composition

**Quick guide for migrating your custom trainers to the new composition-based design**

---

## Table of Contents
1. [Why Migrate?](#why-migrate)
2. [Quick Start](#quick-start)
3. [Step-by-Step Migration](#step-by-step-migration)
4. [Common Patterns](#common-patterns)
5. [Examples by Algorithm](#examples-by-algorithm)
6. [Troubleshooting](#troubleshooting)

---

## Why Migrate?

### Benefits of the New Design

- **Less Code**: 50-90% reduction in trainer code
- **More Flexible**: Mix and match strategies easily
- **Easier to Test**: Test strategies independently
- **Better Maintained**: Bug fixes benefit everyone
- **Clearer Intent**: Strategy names explain what they do

### What Changed?

**Before**: Override methods in `basic.Trainer`
```python
class MyTrainer(basic.Trainer):
    def get_loss_criterion(self):
        return custom_loss

    def train_step_end(self, config, batch, loss):
        # Custom logic
        pass
```

**After**: Inject strategies into `ComposableTrainer`
```python
class MyTrainer(ComposableTrainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=MyLossStrategy(),
            model_update_strategy=MyUpdateStrategy()
        )
```

---

## Quick Start

### 1. Identify Your Algorithm

Determine which strategies your algorithm needs:

- **Loss Modification?** → Use `LossCriterionStrategy`
- **Custom Optimizer?** → Use `OptimizerStrategy`
- **Different Training Step?** → Use `TrainingStepStrategy`
- **State Management?** → Use `ModelUpdateStrategy`
- **LR Scheduling?** → Use `LRSchedulerStrategy`
- **Custom Data Loading?** → Use `DataLoaderStrategy`

### 2. Check If Strategy Exists

Look in `plato/trainers/strategies/implementations/`:

```python
from plato.trainers.strategies.algorithms import (
    FedProxLossStrategy,
    SCAFFOLDUpdateStrategy,
    FedDynLossStrategy,
    LGFedAvgStepStrategy,
    FedMosOptimizerStrategy,
    FedPerUpdateStrategy,
    FedRepUpdateStrategy,
    APFLUpdateStrategy,
    DittoUpdateStrategy,
    # ... and more
)
```

### 3. Use Existing Strategy or Create New One

**If strategy exists**:
```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import FedProxLossStrategy

class MyTrainer(ComposableTrainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=FedProxLossStrategy(mu=0.01)
        )
```

**If you need custom strategy**:
```python
from plato.trainers.strategies.base import LossCriterionStrategy

class MyCustomLossStrategy(LossCriterionStrategy):
    def compute_loss(self, outputs, labels, context):
        # Your custom logic
        return loss

class MyTrainer(ComposableTrainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=MyCustomLossStrategy()
        )
```

---

## Step-by-Step Migration

### Step 1: Analyze Your Current Trainer

Identify which methods you've overridden:

```python
class MyTrainer(basic.Trainer):
    def get_loss_criterion(self):           # → LossCriterionStrategy
        pass

    def get_optimizer(self, model):          # → OptimizerStrategy
        pass

    def perform_forward_and_backward_passes(...):  # → TrainingStepStrategy
        pass

    def train_run_start(self, config):       # → ModelUpdateStrategy.on_train_start
        pass

    def train_step_end(self, config, ...):   # → ModelUpdateStrategy.after_step
        pass

    def train_run_end(self, config):         # → ModelUpdateStrategy.on_train_end
        pass
```

### Step 2: Map Methods to Strategies

| Old Method | New Strategy | Strategy Method |
|------------|--------------|-----------------|
| `get_loss_criterion()` | `LossCriterionStrategy` | `compute_loss()` |
| `get_optimizer()` | `OptimizerStrategy` | `create_optimizer()` |
| `perform_forward_and_backward_passes()` | `TrainingStepStrategy` | `training_step()` |
| `train_run_start()` | `ModelUpdateStrategy` | `on_train_start()` |
| `train_step_end()` | `ModelUpdateStrategy` | `after_step()` |
| `train_run_end()` | `ModelUpdateStrategy` | `on_train_end()` |
| `get_lr_scheduler()` | `LRSchedulerStrategy` | `create_scheduler()` |

### Step 3: Create Strategy Classes

Extract your custom logic into strategy classes:

```python
from plato.trainers.strategies.base import LossCriterionStrategy, ModelUpdateStrategy

class MyLossStrategy(LossCriterionStrategy):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def compute_loss(self, outputs, labels, context):
        # Extract logic from get_loss_criterion()
        base_loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        custom_term = self.param1 * compute_something(context.model)
        return base_loss + custom_term

class MyUpdateStrategy(ModelUpdateStrategy):
    def __init__(self):
        self.state = None

    def on_train_start(self, context):
        # Extract logic from train_run_start()
        self.state = initialize_state(context.model)

    def after_step(self, context):
        # Extract logic from train_step_end()
        update_state(self.state, context.model)

    def on_train_end(self, context):
        # Extract logic from train_run_end()
        save_state(self.state)
```

### Step 4: Update Trainer Class

Replace the old trainer with a simple composition:

```python
from plato.trainers.composable import ComposableTrainer

class MyTrainer(ComposableTrainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=MyLossStrategy(param1=0.01, param2=0.5),
            model_update_strategy=MyUpdateStrategy()
        )
```

### Step 5: Test

Run your tests to ensure everything works:

```bash
python -m pytest tests/test_my_trainer.py
```

---

## Common Patterns

### Pattern 1: Simple Loss Modification

**Before**:
```python
class MyTrainer(basic.Trainer):
    def get_loss_criterion(self):
        return lambda outputs, labels: custom_loss(outputs, labels)
```

**After**:
```python
class MyLossStrategy(LossCriterionStrategy):
    def compute_loss(self, outputs, labels, context):
        return custom_loss(outputs, labels)

class MyTrainer(ComposableTrainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=MyLossStrategy()
        )
```

### Pattern 2: State Management

**Before**:
```python
class MyTrainer(basic.Trainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)
        self.my_state = None

    def train_run_start(self, config):
        super().train_run_start(config)
        self.my_state = initialize()

    def train_step_end(self, config, batch, loss):
        update_state(self.my_state)

    def train_run_end(self, config):
        save_state(self.my_state)
        super().train_run_end(config)
```

**After**:
```python
class MyUpdateStrategy(ModelUpdateStrategy):
    def __init__(self):
        self.my_state = None

    def on_train_start(self, context):
        self.my_state = initialize()

    def after_step(self, context):
        update_state(self.my_state)

    def on_train_end(self, context):
        save_state(self.my_state)

class MyTrainer(ComposableTrainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            model_update_strategy=MyUpdateStrategy()
        )
```

### Pattern 3: Custom Training Step

**Before**:
```python
class MyTrainer(basic.Trainer):
    def perform_forward_and_backward_passes(self, config, examples, labels):
        self.optimizer.zero_grad()
        outputs = self.model(examples)
        loss = self._loss_criterion(outputs, labels)

        # Custom logic here
        custom_step(self.model, loss)

        self.optimizer.step()
        return loss
```

**After**:
```python
class MyStepStrategy(TrainingStepStrategy):
    def training_step(self, model, optimizer, examples, labels,
                      loss_criterion, context):
        optimizer.zero_grad()
        outputs = model(examples)
        loss = loss_criterion(outputs, labels)

        # Custom logic here
        custom_step(model, loss)

        optimizer.step()
        return loss

class MyTrainer(ComposableTrainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            training_step_strategy=MyStepStrategy()
        )
```

### Pattern 4: Combining Multiple Strategies

**Before** (Complex inheritance):
```python
class MyComplexTrainer(basic.Trainer):
    def get_loss_criterion(self):
        # FedProx-like logic
        pass

    def train_run_start(self, config):
        # SCAFFOLD-like logic
        pass

    def train_step_end(self, config, batch, loss):
        # SCAFFOLD-like logic
        pass
```

**After** (Simple composition):
```python
class MyTrainer(ComposableTrainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=FedProxLossStrategy(mu=0.01),
            model_update_strategy=SCAFFOLDUpdateStrategy()
        )
```

---

## Examples by Algorithm

### FedProx

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import FedProxLossStrategy

class FedProxTrainer(ComposableTrainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=FedProxLossStrategy(mu=0.01)
        )
```

### SCAFFOLD

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import SCAFFOLDUpdateStrategy

class SCAFFOLDTrainer(ComposableTrainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            model_update_strategy=SCAFFOLDUpdateStrategy()
        )
        self.additional_data = None

    def set_client_id(self, client_id):
        super().set_client_id(client_id)
        if self.additional_data is not None:
            self.training_context.state["server_control_variate"] = self.additional_data
```

### FedDyn

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import (
    FedDynLossStrategy,
    FedDynUpdateStrategy
)

class FedDynTrainer(ComposableTrainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=FedDynLossStrategy(alpha=0.01),
            model_update_strategy=FedDynUpdateStrategy()
        )
```

### LG-FedAvg

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import LGFedAvgStepStrategy

class LGFedAvgTrainer(ComposableTrainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            training_step_strategy=LGFedAvgStepStrategy(
                global_layer_names=['conv1', 'conv2', 'fc1'],
                local_layer_names=['fc2']
            )
        )
```

### APFL

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import (
    APFLUpdateStrategy,
    APFLStepStrategy
)

class APFLTrainer(ComposableTrainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            model_update_strategy=APFLUpdateStrategy(alpha=0.5),
            training_step_strategy=APFLStepStrategy()
        )
```

### Ditto

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import DittoUpdateStrategy

class DittoTrainer(ComposableTrainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            model_update_strategy=DittoUpdateStrategy(
                ditto_lambda=0.1,
                personalization_epochs=5
            )
        )
```

---

## Troubleshooting

### Issue: "AttributeError: 'ComposableTrainer' object has no attribute 'X'"

**Problem**: Trying to access attributes from the old trainer.

**Solution**: Strategies manage their own state. Access via `context.state`:
```python
# Before
value = self.my_attribute

# After (in strategy)
value = context.state.get('my_attribute')
```

### Issue: "My custom logic isn't being called"

**Problem**: Logic in wrong lifecycle method.

**Solution**: Check the strategy interface:
- `setup()`: Called once at initialization
- `on_train_start()`: Called at start of each round
- `before_step()`: Called before each training step
- `after_step()`: Called after each training step
- `on_train_end()`: Called at end of each round
- `teardown()`: Called once at end of all training

### Issue: "Strategies don't work together"

**Problem**: Incompatible strategies.

**Solution**: Check compatibility matrix in `ALGORITHM_STRATEGIES_QUICK_REFERENCE.md`:
- APFL requires both `APFLUpdateStrategy` and `APFLStepStrategy`
- FedDyn requires both `FedDynLossStrategy` and `FedDynUpdateStrategy`
- Most other strategies can be combined freely

### Issue: "Config parameters not being read"

**Problem**: Using explicit strategy instead of `*FromConfig` variant.

**Solution**: Use the config-based variant:
```python
# Instead of
loss_strategy=FedProxLossStrategy(mu=0.01)

# Use
loss_strategy=FedProxLossStrategyFromConfig()

# And set in config.yml:
# clients:
#   proximal_term_penalty_constant: 0.01
```

### Issue: "State not persisting between rounds"

**Problem**: Not saving state to disk.

**Solution**: Implement persistence in strategy:
```python
class MyUpdateStrategy(ModelUpdateStrategy):
    def on_train_end(self, context):
        # Save state
        torch.save(self.state, f"state_{context.client_id}.pth")

    def on_train_start(self, context):
        # Load state
        if os.path.exists(f"state_{context.client_id}.pth"):
            self.state = torch.load(f"state_{context.client_id}.pth")
```

### Issue: "Performance degradation after migration"

**Problem**: Extra overhead from strategy indirection.

**Solution**:
1. Profile to find bottleneck
2. Move performance-critical code to compiled functions
3. Use `@torch.jit.script` for hot paths
4. Strategy overhead is typically <1% - if seeing more, something else is wrong

---

## Best Practices

### 1. Use Config-Based Variants in Production

```python
# Development
trainer = ComposableTrainer(
    loss_strategy=FedProxLossStrategy(mu=0.01)
)

# Production
trainer = ComposableTrainer(
    loss_strategy=FedProxLossStrategyFromConfig()
)
```

### 2. Document Your Strategies

```python
class MyStrategy(LossCriterionStrategy):
    """
    My custom loss strategy.

    Reference:
    Author et al., "Paper Title", Conference Year.

    Description:
    This strategy does X by computing Y...

    Args:
        param1: Description of param1
        param2: Description of param2

    Example:
        >>> strategy = MyStrategy(param1=0.01)
        >>> trainer = ComposableTrainer(loss_strategy=strategy)
    """
```

### 3. Test Strategies Independently

```python
def test_my_strategy():
    strategy = MyStrategy(param=0.01)
    context = TrainingContext()
    context.model = create_test_model()

    strategy.setup(context)
    result = strategy.compute_loss(outputs, labels, context)

    assert result > 0
```

### 4. Share State via Context

```python
class ProducerStrategy(ModelUpdateStrategy):
    def on_train_start(self, context):
        context.state['shared_data'] = compute_data()

class ConsumerStrategy(LossCriterionStrategy):
    def compute_loss(self, outputs, labels, context):
        shared_data = context.state.get('shared_data')
        return compute_loss_with(shared_data)
```

### 5. Keep Trainers Simple

```python
# Good: Trainer just injects strategies
class MyTrainer(ComposableTrainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=MyLossStrategy(),
            model_update_strategy=MyUpdateStrategy()
        )

# Bad: Trainer has additional logic
class MyTrainer(ComposableTrainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(...)
        self.extra_state = ...  # Move to strategy!

    def custom_method(self):  # Move to strategy!
        pass
```

---

## Getting Help

- **Documentation**: See `docs/trainer-redesign/` for detailed guides
- **Examples**: Check migrated examples in `examples/` directory
- **Reference**: `ALGORITHM_STRATEGIES_QUICK_REFERENCE.md` for quick lookup
- **Issues**: File issues on GitHub with `[Migration]` prefix

---

## Summary

**Migration Steps**:
1. Identify which methods you override
2. Map methods to strategy types
3. Create strategy classes with your logic
4. Inject strategies into ComposableTrainer
5. Test and validate

**Key Benefits**:
- Less code (50-90% reduction)
- More flexible (mix strategies)
- Easier to test (test in isolation)
- Better maintained (shared strategies)

**Common Pitfalls**:
- Don't add logic to trainer class - use strategies
- Don't forget to save/load state in strategies
- Don't mix incompatible strategies
- Do use `*FromConfig` variants for production

---

**Happy Migrating! 🚀**
