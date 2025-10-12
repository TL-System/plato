# Algorithm Strategies Quick Reference Guide

**Quick reference for using algorithm-specific strategies in Plato**

---

## Table of Contents
1. [FedProx](#fedprox)
2. [SCAFFOLD](#scaffold)
3. [FedDyn](#feddyn)
4. [LG-FedAvg](#lg-fedavg)
5. [FedMos](#fedmos)
6. [FedPer](#fedper)
7. [FedRep](#fedrep)
8. [APFL](#apfl)
9. [Ditto](#ditto)
10. [Combining Strategies](#combining-strategies)

---

## FedProx

**Purpose**: Handle system heterogeneity with proximal term regularization

**Strategy Type**: `LossCriterionStrategy`

### Basic Usage
```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import FedProxLossStrategy

trainer = ComposableTrainer(
    loss_strategy=FedProxLossStrategy(mu=0.01)
)
```

### Config-Based Usage
```yaml
# config.yml
clients:
  proximal_term_penalty_constant: 0.01
```

```python
from plato.trainers.strategies.algorithms import FedProxLossStrategyFromConfig

trainer = ComposableTrainer(
    loss_strategy=FedProxLossStrategyFromConfig()
)
```

### Parameters
- `mu` (float): Proximal term coefficient (default: 0.01)
- `base_loss_fn` (callable): Base loss function (default: CrossEntropyLoss)
- `norm_type` (str): 'l2' or 'l1' (default: 'l2')

### When to Use
- Clients have heterogeneous computational resources
- High variance in client updates
- Non-IID data distribution

---

## SCAFFOLD

**Purpose**: Reduce client drift with control variates

**Strategy Type**: `ModelUpdateStrategy`

### Basic Usage
```python
from plato.trainers.strategies.algorithms import SCAFFOLDUpdateStrategy

trainer = ComposableTrainer(
    model_update_strategy=SCAFFOLDUpdateStrategy()
)
```

### Server-Side Setup
```python
# Server must send control variate to clients
context.state['server_control_variate'] = server_control_variate

# After training, receive delta from client
payload = trainer.get_update_payload(context)
delta = payload['control_variate_delta']

# Update server control variate
server_control_variate += (1/K) * sum(deltas)
```

### Parameters
- `save_path` (str): Custom path for saving control variates (optional)

### When to Use
- Severe client drift issues
- Non-IID data with large variance
- Need variance reduction in gradients

### Variants
- `SCAFFOLDUpdateStrategy`: Option 2 from paper (uses model difference)
- `SCAFFOLDUpdateStrategyV2`: Option 1 from paper (accumulates updates)

---

## FedDyn

**Purpose**: Dynamic regularization for federated learning

**Strategy Type**: `LossCriterionStrategy` + `ModelUpdateStrategy`

### Basic Usage
```python
from plato.trainers.strategies.algorithms import (
    FedDynLossStrategy,
    FedDynUpdateStrategy
)

trainer = ComposableTrainer(
    loss_strategy=FedDynLossStrategy(alpha=0.01),
    model_update_strategy=FedDynUpdateStrategy()
)
```

### Config-Based Usage
```yaml
# config.yml
algorithm:
  alpha_coef: 0.01
```

```python
from plato.trainers.strategies.algorithms import (
    FedDynLossStrategyFromConfig,
    FedDynUpdateStrategy
)

trainer = ComposableTrainer(
    loss_strategy=FedDynLossStrategyFromConfig(),
    model_update_strategy=FedDynUpdateStrategy()
)
```

### Parameters
- `alpha` (float): Regularization coefficient (default: 0.01)
- `adaptive_alpha` (bool): Scale by client data weight (default: True)

### When to Use
- Non-IID data distribution
- Need adaptive regularization
- Alternative to FedProx

---

## LG-FedAvg

**Purpose**: Personalized FL with local and global layers

**Strategy Type**: `TrainingStepStrategy`

### Basic Usage
```python
from plato.trainers.strategies.algorithms import LGFedAvgStepStrategy

trainer = ComposableTrainer(
    training_step_strategy=LGFedAvgStepStrategy(
        global_layer_names=['conv1', 'conv2', 'fc1'],
        local_layer_names=['fc2']
    )
)
```

### Config-Based Usage
```yaml
# config.yml
algorithm:
  global_layer_names:
    - conv1
    - conv2
    - fc1
  local_layer_names:
    - fc2
  train_local_first: true
```

```python
from plato.trainers.strategies.algorithms import LGFedAvgStepStrategyFromConfig

trainer = ComposableTrainer(
    training_step_strategy=LGFedAvgStepStrategyFromConfig()
)
```

### Auto Layer Detection
```python
from plato.trainers.strategies.algorithms import LGFedAvgStepStrategyAuto

# Automatically treat last N layers as local
trainer = ComposableTrainer(
    training_step_strategy=LGFedAvgStepStrategyAuto(num_local_layers=1)
)
```

### Parameters
- `global_layer_names` (list): Patterns for global layers
- `local_layer_names` (list): Patterns for local layers
- `train_local_first` (bool): Train local first (default: True)

### When to Use
- Need personalization
- Different layers serve different purposes
- Want to keep some layers private

---

## FedMos

**Purpose**: Double momentum for stable training

**Strategy Type**: `OptimizerStrategy` + `ModelUpdateStrategy`

### Basic Usage
```python
from plato.trainers.strategies.algorithms import (
    FedMosOptimizerStrategy,
    FedMosUpdateStrategy
)

trainer = ComposableTrainer(
    optimizer_strategy=FedMosOptimizerStrategy(lr=0.01, a=0.9, mu=0.9),
    model_update_strategy=FedMosUpdateStrategy()
)
```

### Config-Based Usage
```yaml
# config.yml
algorithm:
  a: 0.9    # Local momentum
  mu: 0.9   # Global momentum
parameters:
  optimizer:
    lr: 0.01
```

```python
from plato.trainers.strategies.algorithms import (
    FedMosOptimizerStrategyFromConfig,
    FedMosUpdateStrategy
)

trainer = ComposableTrainer(
    optimizer_strategy=FedMosOptimizerStrategyFromConfig(),
    model_update_strategy=FedMosUpdateStrategy()
)
```

### Parameters
- `lr` (float): Learning rate (default: 0.01)
- `a` (float): Local momentum coefficient (default: 0.9)
- `mu` (float): Global momentum coefficient (default: 0.9)
- `weight_decay` (float): Weight decay (default: 0)

### When to Use
- Need momentum-based optimization
- Client drift is an issue
- Want to balance local and global updates

---

## FedPer

**Purpose**: Simple personalization with layer freezing

**Strategy Type**: `ModelUpdateStrategy`

### Basic Usage
```python
from plato.trainers.strategies.algorithms import FedPerUpdateStrategy

trainer = ComposableTrainer(
    model_update_strategy=FedPerUpdateStrategy(
        global_layer_names=['conv1', 'conv2', 'fc1']
    )
)
```

### Config-Based Usage
```yaml
# config.yml
algorithm:
  global_layer_names:
    - conv1
    - conv2
    - fc1
trainer:
  rounds: 100  # Personalization starts after this
```

```python
from plato.trainers.strategies.algorithms import FedPerUpdateStrategyFromConfig

trainer = ComposableTrainer(
    model_update_strategy=FedPerUpdateStrategyFromConfig()
)
```

### Parameters
- `global_layer_names` (list): Patterns for global layers
- `personalization_rounds` (int): Number of personalization rounds (optional)

### When to Use
- Simple personalization needs
- Want to freeze global layers in final rounds
- Less complex than FedRep

---

## FedRep

**Purpose**: Representation learning with alternating training

**Strategy Type**: `ModelUpdateStrategy`

### Basic Usage
```python
from plato.trainers.strategies.algorithms import FedRepUpdateStrategy

trainer = ComposableTrainer(
    model_update_strategy=FedRepUpdateStrategy(
        global_layer_names=['conv1', 'conv2'],
        local_layer_names=['fc1', 'fc2'],
        local_epochs=2
    )
)
```

### Config-Based Usage
```yaml
# config.yml
algorithm:
  global_layer_names:
    - conv1
    - conv2
  local_layer_names:
    - fc1
    - fc2
  local_epochs: 2
```

```python
from plato.trainers.strategies.algorithms import FedRepUpdateStrategyFromConfig

trainer = ComposableTrainer(
    model_update_strategy=FedRepUpdateStrategyFromConfig()
)
```

### Parameters
- `global_layer_names` (list): Patterns for global layers
- `local_layer_names` (list): Patterns for local layers
- `local_epochs` (int): Epochs for local layer training (default: 1)

### Training Flow
1. Train local layers for `local_epochs` epochs
2. Train global layers for remaining epochs
3. Repeat each round

### When to Use
- Need better representation learning
- Want separation of features and classifier
- More sophisticated than FedPer

---

## APFL

**Purpose**: Adaptive personalization with dual models

**Strategy Type**: `ModelUpdateStrategy` + `TrainingStepStrategy`

### Basic Usage
```python
from plato.trainers.strategies.algorithms import (
    APFLUpdateStrategy,
    APFLStepStrategy
)

trainer = ComposableTrainer(
    model_update_strategy=APFLUpdateStrategy(alpha=0.5),
    training_step_strategy=APFLStepStrategy()
)
```

### Config-Based Usage
```yaml
# config.yml
algorithm:
  alpha: 0.5
  adaptive_alpha: true
  alpha_lr: 0.01
```

```python
from plato.trainers.strategies.algorithms import (
    APFLUpdateStrategyFromConfig,
    APFLStepStrategy
)

trainer = ComposableTrainer(
    model_update_strategy=APFLUpdateStrategyFromConfig(),
    training_step_strategy=APFLStepStrategy()
)
```

### Parameters
- `alpha` (float): Mixing parameter, 0=fully personalized, 1=fully global (default: 0.5)
- `adaptive_alpha` (bool): Learn α adaptively (default: True)
- `alpha_lr` (float): Learning rate for α (default: 0.01)

### How It Works
- Maintains global model (w) and personalized model (v)
- Output = α * v + (1 - α) * w
- α is learned adaptively per client

### When to Use
- Want adaptive personalization
- Each client needs different global/local balance
- Willing to maintain two models per client

---

## Ditto

**Purpose**: Fair personalization with post-training

**Strategy Type**: `ModelUpdateStrategy`

### Basic Usage
```python
from plato.trainers.strategies.algorithms import DittoUpdateStrategy

trainer = ComposableTrainer(
    model_update_strategy=DittoUpdateStrategy(
        ditto_lambda=0.1,
        personalization_epochs=5
    )
)
```

### Config-Based Usage
```yaml
# config.yml
algorithm:
  ditto_lambda: 0.1
  personalization_epochs: 5
```

```python
from plato.trainers.strategies.algorithms import DittoUpdateStrategyFromConfig

trainer = ComposableTrainer(
    model_update_strategy=DittoUpdateStrategyFromConfig()
)
```

### Parameters
- `ditto_lambda` (float): Regularization coefficient (default: 0.1)
- `personalization_epochs` (int): Epochs for personalization (default: 5)

### Training Flow
1. Train global model normally
2. After global training completes, train personalized model
3. Personalized model regularized towards global: min F(v) + λ/2 ||v - w||²
4. Send only global model to server
5. Use personalized model for inference

### When to Use
- Want fairness guarantees
- Need personalization without changing main training
- Can afford extra training after global model updates

---

## Combining Strategies

### Example 1: FedProx + FedMos
```python
from plato.trainers.strategies.algorithms import (
    FedProxLossStrategy,
    FedMosOptimizerStrategy,
    FedMosUpdateStrategy
)

trainer = ComposableTrainer(
    loss_strategy=FedProxLossStrategy(mu=0.01),
    optimizer_strategy=FedMosOptimizerStrategy(lr=0.01, a=0.9, mu=0.9),
    model_update_strategy=FedMosUpdateStrategy()
)
```

### Example 2: FedDyn + Custom Optimizer
```python
from plato.trainers.strategies.algorithms import (
    FedDynLossStrategy,
    FedDynUpdateStrategy
)
from plato.trainers.strategies.optimizer import AdamOptimizerStrategy

trainer = ComposableTrainer(
    loss_strategy=FedDynLossStrategy(alpha=0.01),
    optimizer_strategy=AdamOptimizerStrategy(lr=0.001),
    model_update_strategy=FedDynUpdateStrategy()
)
```

### Example 3: LG-FedAvg + FedProx
```python
from plato.trainers.strategies.algorithms import (
    FedProxLossStrategy,
    LGFedAvgStepStrategy
)

trainer = ComposableTrainer(
    loss_strategy=FedProxLossStrategy(mu=0.01),
    training_step_strategy=LGFedAvgStepStrategy(
        global_layer_names=['conv1', 'conv2'],
        local_layer_names=['fc1', 'fc2']
    )
)
```

---

## Strategy Compatibility Matrix

| Strategy | Compatible With | Notes |
|----------|----------------|-------|
| FedProx | All | Loss strategy, can combine with any |
| SCAFFOLD | All except APFL | Update strategy, manages control variates |
| FedDyn | All except APFL | Loss + Update, pair both strategies |
| LG-FedAvg | All except APFL | Step strategy, controls training flow |
| FedMos | All except APFL | Optimizer + Update, pair both strategies |
| FedPer | All except APFL, FedRep | Update strategy, simple personalization |
| FedRep | All except APFL, FedPer | Update strategy, advanced personalization |
| APFL | None | Requires both its strategies, self-contained |
| Ditto | All | Update strategy, post-training personalization |

---

## Common Pitfalls

### 1. Forgetting Paired Strategies
❌ **Wrong**: Only using FedDynLossStrategy
```python
trainer = ComposableTrainer(
    loss_strategy=FedDynLossStrategy(alpha=0.01)
)
# Missing FedDynUpdateStrategy!
```

✅ **Correct**: Using both
```python
trainer = ComposableTrainer(
    loss_strategy=FedDynLossStrategy(alpha=0.01),
    model_update_strategy=FedDynUpdateStrategy()
)
```

### 2. Wrong Layer Names
❌ **Wrong**: Using incorrect patterns
```python
LGFedAvgStepStrategy(
    global_layer_names=['fc.weight'],  # Too specific
    local_layer_names=['classifier']
)
```

✅ **Correct**: Using partial matches
```python
LGFedAvgStepStrategy(
    global_layer_names=['conv', 'fc1'],  # Partial match works
    local_layer_names=['fc2']
)
```

### 3. Missing Server-Side Logic
❌ **Wrong**: Using SCAFFOLD without server support
```python
# Client only
trainer = ComposableTrainer(
    model_update_strategy=SCAFFOLDUpdateStrategy()
)
```

✅ **Correct**: Implementing both client and server
```python
# Client
trainer = ComposableTrainer(
    model_update_strategy=SCAFFOLDUpdateStrategy()
)

# Server
context.state['server_control_variate'] = server_cv
# ... aggregate deltas and update server_cv
```

### 4. Combining Incompatible Strategies
❌ **Wrong**: Using multiple personalization strategies
```python
trainer = ComposableTrainer(
    model_update_strategy=FedPerUpdateStrategy(...),  # Both try to
    training_step_strategy=APFLStepStrategy()         # manage personalization
)
```

✅ **Correct**: Choose one personalization approach
```python
trainer = ComposableTrainer(
    model_update_strategy=FedPerUpdateStrategy(...)
)
```

---

## Performance Tips

### 1. Use Config-Based Variants in Production
```python
# Development: explicit parameters
trainer = ComposableTrainer(
    loss_strategy=FedProxLossStrategy(mu=0.01)
)

# Production: config-based
trainer = ComposableTrainer(
    loss_strategy=FedProxLossStrategyFromConfig()
)
```

### 2. Save State Properly
All strategies with state (SCAFFOLD, FedDyn, APFL, Ditto) save to disk automatically.
Ensure the model_path in config is writable:
```yaml
params:
  model_path: "./models"  # Ensure this exists and is writable
```

### 3. Monitor Memory Usage
Strategies with dual models (APFL, Ditto) use more memory:
- Move models to CPU when not in use
- Use smaller personalization epochs
- Consider batch size adjustments

### 4. Use Auto Variants for Quick Prototyping
```python
# Instead of specifying all layers
trainer = ComposableTrainer(
    training_step_strategy=LGFedAvgStepStrategyAuto(num_local_layers=1)
)
```

---

## Debugging Tips

### 1. Enable Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### 2. Check Context State
```python
def on_train_start(self, context):
    print(f"Context state: {context.state.keys()}")
```

### 3. Validate Layer Names
```python
# Print model parameter names
for name, _ in model.named_parameters():
    print(name)
```

### 4. Test with Small Model First
```python
# Use a simple model for initial testing
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)
```

---

## Quick Selection Guide

**Choose based on your needs:**

| Need | Algorithm | Complexity | Overhead |
|------|-----------|------------|----------|
| Handle heterogeneity | FedProx | Low | Minimal |
| Reduce drift | SCAFFOLD | High | Moderate |
| Dynamic regularization | FedDyn | Medium | Low |
| Simple personalization | FedPer | Low | Minimal |
| Layer-wise personalization | LG-FedAvg | Medium | Low |
| Representation learning | FedRep | Medium | Low |
| Adaptive personalization | APFL | High | High (2 models) |
| Fair personalization | Ditto | Medium | Moderate |
| Stable optimization | FedMos | Medium | Low |

---

## Getting Help

- **Documentation**: See `PHASE3_COMPLETION.md` for detailed implementation notes
- **Examples**: Check `examples/composable_trainer_example.py`
- **Issues**: File issues on GitHub with `[Strategy]` prefix
- **Papers**: All strategies include paper references in docstrings

---

**Last Updated**: Phase 3 Completion  
**Version**: 1.0