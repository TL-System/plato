# Trainer Refactoring Quick Reference Card

## 🚀 Getting Started (30 seconds)

### Use ComposableTrainer with Defaults
```python
from plato.trainers.composable import ComposableTrainer

trainer = ComposableTrainer(model=my_model)
# That's it! Uses all sensible defaults
```

### Customize Loss
```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies import CrossEntropyLossStrategy

trainer = ComposableTrainer(
    model=my_model,
    loss_strategy=CrossEntropyLossStrategy(label_smoothing=0.1)
)
```

### Multiple Strategies
```python
from plato.trainers.strategies import (
    AdamOptimizerStrategy,
    CosineAnnealingLRSchedulerStrategy,
)

trainer = ComposableTrainer(
    model=my_model,
    optimizer_strategy=AdamOptimizerStrategy(lr=0.001),
    lr_scheduler_strategy=CosineAnnealingLRSchedulerStrategy(T_max=50),
)
```

---

## 📦 What's Available

### 6 Strategy Types (40 implementations)

| Strategy Type | Purpose | Implementations |
|---------------|---------|-----------------|
| **LossCriterionStrategy** | Compute loss | CrossEntropy, MSE, BCE, NLL, Composite, L2Reg (7) |
| **OptimizerStrategy** | Create optimizer | SGD, Adam, AdamW, RMSprop, ParameterGroup (7) |
| **TrainingStepStrategy** | Training step logic | Default, GradAccum, MixedPrecision, GradClip (7) |
| **LRSchedulerStrategy** | LR scheduling | Cosine, Step, MultiStep, Warmup, Exponential (11) |
| **ModelUpdateStrategy** | State management | NoOp, StateTracking, Composite (3) |
| **DataLoaderStrategy** | Data loading | Default, Prefetch, CustomCollate, DynamicBatch (5) |

---

## 🔧 Common Use Cases

### Case 1: Change Loss Function
```python
from plato.trainers.strategies import MSELossStrategy

trainer = ComposableTrainer(
    model=my_model,
    loss_strategy=MSELossStrategy()  # For regression
)
```

### Case 2: Use Adam Optimizer
```python
from plato.trainers.strategies import AdamOptimizerStrategy

trainer = ComposableTrainer(
    model=my_model,
    optimizer_strategy=AdamOptimizerStrategy(lr=0.001, weight_decay=0.01)
)
```

### Case 3: Gradient Accumulation
```python
from plato.trainers.strategies import GradientAccumulationStepStrategy

trainer = ComposableTrainer(
    model=my_model,
    training_step_strategy=GradientAccumulationStepStrategy(
        accumulation_steps=4  # Effectively 4x batch size
    )
)
```

### Case 4: Mixed Precision Training
```python
from plato.trainers.strategies import MixedPrecisionStepStrategy

trainer = ComposableTrainer(
    model=my_model,
    training_step_strategy=MixedPrecisionStepStrategy(enabled=True)
)
```

### Case 5: Cosine Annealing LR
```python
from plato.trainers.strategies import CosineAnnealingLRSchedulerStrategy

trainer = ComposableTrainer(
    model=my_model,
    lr_scheduler_strategy=CosineAnnealingLRSchedulerStrategy(T_max=50)
)
```

### Case 6: Combine Multiple Losses
```python
from plato.trainers.strategies import (
    CompositeLossStrategy,
    CrossEntropyLossStrategy,
    L2RegularizationStrategy,
)

composite_loss = CompositeLossStrategy([
    (CrossEntropyLossStrategy(), 1.0),
    (L2RegularizationStrategy(weight=0.01), 1.0),
])

trainer = ComposableTrainer(model=my_model, loss_strategy=composite_loss)
```

---

## 🎨 Create Custom Strategy

### Step 1: Choose Strategy Type
Decide which aspect to customize:
- Loss? → `LossCriterionStrategy`
- Optimizer? → `OptimizerStrategy`
- Training step? → `TrainingStepStrategy`
- LR schedule? → `LRSchedulerStrategy`
- State management? → `ModelUpdateStrategy`
- Data loading? → `DataLoaderStrategy`

### Step 2: Implement
```python
from plato.trainers.strategies.base import LossCriterionStrategy, TrainingContext
import torch
import torch.nn as nn

class MyCustomLoss(LossCriterionStrategy):
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self._criterion = None
    
    def setup(self, context: TrainingContext):
        """Called once at initialization"""
        self._criterion = nn.CrossEntropyLoss()
    
    def compute_loss(self, outputs, labels, context):
        """Called for each batch"""
        ce_loss = self._criterion(outputs, labels)
        reg_term = self.alpha * torch.norm(outputs)
        return ce_loss + reg_term
```

### Step 3: Use It
```python
trainer = ComposableTrainer(
    model=my_model,
    loss_strategy=MyCustomLoss(alpha=0.3)
)
```

---

## 📖 Import Cheat Sheet

### Import Trainer
```python
from plato.trainers.composable import ComposableTrainer
```

### Import Common Strategies
```python
from plato.trainers.strategies import (
    # Loss
    CrossEntropyLossStrategy,
    MSELossStrategy,
    
    # Optimizer
    AdamOptimizerStrategy,
    SGDOptimizerStrategy,
    
    # Training Step
    DefaultTrainingStepStrategy,
    GradientAccumulationStepStrategy,
    MixedPrecisionStepStrategy,
    
    # LR Scheduler
    CosineAnnealingLRSchedulerStrategy,
    StepLRSchedulerStrategy,
    
    # Model Update
    NoOpUpdateStrategy,
    
    # Data Loader
    DefaultDataLoaderStrategy,
)
```

### Import Base Classes (for custom strategies)
```python
from plato.trainers.strategies.base import (
    LossCriterionStrategy,
    OptimizerStrategy,
    TrainingStepStrategy,
    LRSchedulerStrategy,
    ModelUpdateStrategy,
    DataLoaderStrategy,
    TrainingContext,
)
```

---

## 🔍 Quick Troubleshooting

### Issue: "Module not found"
**Solution**: Make sure you're in the plato directory and PYTHONPATH is set
```bash
cd /path/to/plato
export PYTHONPATH=/path/to/plato:$PYTHONPATH
```

### Issue: "No module named 'torch'"
**Solution**: Install dependencies
```bash
pip install torch
```

### Issue: Loss not decreasing
**Check**:
1. Learning rate too high/low?
2. Batch size appropriate?
3. Model architecture correct?
4. Data normalized?

### Issue: Want to combine algorithms (e.g., FedProx + SCAFFOLD)
**Solution**: Phase 3 coming soon! For now, create custom strategies

---

## 📊 Strategy Selection Guide

| Scenario | Recommended Strategy |
|----------|---------------------|
| Classification | `CrossEntropyLossStrategy` |
| Regression | `MSELossStrategy` |
| Binary classification | `BCEWithLogitsLossStrategy` |
| Fast convergence | `AdamOptimizerStrategy` or `AdamWOptimizerStrategy` |
| Limited GPU memory | `GradientAccumulationStepStrategy` |
| Have GPU | `MixedPrecisionStepStrategy` |
| Gradient explosion | `GradientClippingStepStrategy` |
| Fine-tuning | `CosineAnnealingLRSchedulerStrategy` |
| Step decay | `StepLRSchedulerStrategy` |
| Warmup needed | `WarmupSchedulerStrategy` |
| Fast data loading | `PrefetchDataLoaderStrategy` |

---

## 🎯 Complete Example

```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies import (
    CrossEntropyLossStrategy,
    AdamWOptimizerStrategy,
    MixedPrecisionStepStrategy,
    CosineAnnealingLRSchedulerStrategy,
    PrefetchDataLoaderStrategy,
)

# Create trainer with multiple strategies
trainer = ComposableTrainer(
    model=my_cnn_model,
    loss_strategy=CrossEntropyLossStrategy(label_smoothing=0.1),
    optimizer_strategy=AdamWOptimizerStrategy(
        lr=0.001,
        weight_decay=0.01
    ),
    training_step_strategy=MixedPrecisionStepStrategy(enabled=True),
    lr_scheduler_strategy=CosineAnnealingLRSchedulerStrategy(T_max=50),
    data_loader_strategy=PrefetchDataLoaderStrategy(
        prefetch_factor=4,
        num_workers=4
    ),
)

# Use in federated learning
from plato.clients import simple
from plato.servers import fedavg

client = simple.Client(trainer=trainer)
server = fedavg.Server(trainer=trainer)
server.run(client)
```

---

## 📚 Resources

- **Quick Start**: `plato/trainers/strategies/README.md`
- **Examples**: `plato/examples/composable_trainer_example.py`
- **Design**: `TRAINER_REFACTORING_DESIGN.md`
- **API Docs**: Docstrings in each strategy class
- **Tests**: `tests/trainers/test_composable_trainer.py`

---

## ✨ Key Takeaways

1. ✅ **Default strategies work** - No need to specify everything
2. ✅ **Mix and match** - Combine any strategies
3. ✅ **Easy to test** - Unit test each strategy independently
4. ✅ **Type safe** - Full type hints for IDE support
5. ✅ **Well documented** - Every strategy has examples
6. ✅ **Backward compatible** - Existing code still works

---

## 🚦 Status

- ✅ **Phase 1**: Strategy interfaces complete
- ✅ **Phase 2**: ComposableTrainer complete
- ⏳ **Phase 3**: Algorithm strategies (FedProx, SCAFFOLD, etc.) coming soon

---

**Version**: 1.0  
**Last Updated**: 2024  
**Status**: Production Ready ✅