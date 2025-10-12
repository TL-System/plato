# LG-FedAvg Implementation Comparison

## Overview
Comparing the old inheritance-based implementation (main branch) with the new composition-based implementation (trainer-refactor branch) of LG-FedAvg algorithm.

## Paper Reference
**"Think Locally, Act Globally: Federated Learning with Local and Global Representations"**
- Authors: P. Liang et al.
- Conference: NeurIPS 2019
- ArXiv: https://arxiv.org/abs/2001.01523
- Source Code: https://github.com/pliang279/LG-FedAvg

## Algorithm Description (from Paper)
LG-FedAvg divides a neural network model into two parts:
1. **Global layers**: Shared across all clients and aggregated on the server
2. **Local layers**: Kept locally on each client and not shared

During training, LG-FedAvg performs **two forward/backward passes per iteration**:
1. **First pass**: Freeze global layers, train only local layers
2. **Second pass**: Freeze local layers, train only global layers

This allows personalized federated learning where clients learn client-specific representations (local) while benefiting from shared features (global).

---

## Implementation Comparison

### OLD Implementation (main branch - Inheritance-based)

**File**: `examples/personalized_fl/lgfedavg/lgfedavg_trainer.py`

```python
class Trainer(basic.Trainer):
    def perform_forward_and_backward_passes(self, config, examples, labels):
        # Step 1: Train local layers only
        trainer_utils.freeze_model(self.model, Config().algorithm.global_layer_names)
        trainer_utils.activate_model(self.model, Config().algorithm.local_layer_names)
        super().perform_forward_and_backward_passes(config, examples, labels)

        # Step 2: Train global layers only
        trainer_utils.activate_model(self.model, Config().algorithm.global_layer_names)
        trainer_utils.freeze_model(self.model, Config().algorithm.local_layer_names)
        loss = super().perform_forward_and_backward_passes(config, examples, labels)

        return loss
```

**Helper functions from `plato/utils/trainer_utils.py`**:
```python
def freeze_model(model, layer_names=None):
    if layer_names is not None:
        for name, param in model.named_parameters():
            if any(param_name in name for param_name in layer_names):
                param.requires_grad = False

def activate_model(model, layer_names=None):
    if layer_names is not None:
        for name, param in model.named_parameters():
            if any(param_name in name for param_name in layer_names):
                param.requires_grad = True
```

**Base implementation from `plato/trainers/basic.py`**:
```python
def perform_forward_and_backward_passes(self, config, examples, labels):
    self.optimizer.zero_grad()
    outputs = self.model(examples)
    loss = self._loss_criterion(outputs, labels)
    self._loss_tracker.update(loss, labels.size(0))
    loss.backward()
    self.optimizer.step()
    return loss
```

### NEW Implementation (trainer-refactor branch - Composition-based)

**File**: `plato/trainers/strategies/algorithms/lgfedavg_strategy.py`

```python
class LGFedAvgStepStrategy(TrainingStepStrategy):
    def training_step(self, model, optimizer, examples, labels, loss_criterion, context):
        # Determine training order
        if self.train_local_first:
            first_layers = self.local_layer_names
            first_freeze = self.global_layer_names
            second_layers = self.global_layer_names
            second_freeze = self.local_layer_names
        else:
            first_layers = self.global_layer_names
            first_freeze = self.local_layer_names
            second_layers = self.local_layer_names
            second_freeze = self.global_layer_names

        # First pass: Train first set of layers
        self._freeze_layers(model, first_freeze)
        self._activate_layers(model, first_layers)
        
        optimizer.zero_grad()
        outputs = model(examples)
        loss_first = loss_criterion(outputs, labels)
        loss_first.backward()
        optimizer.step()

        # Second pass: Train second set of layers
        self._freeze_layers(model, second_freeze)
        self._activate_layers(model, second_layers)
        
        optimizer.zero_grad()
        outputs = model(examples)
        loss_second = loss_criterion(outputs, labels)
        loss_second.backward()
        optimizer.step()

        # Re-enable all gradients
        self._activate_layers(model, self.global_layer_names)
        self._activate_layers(model, self.local_layer_names)

        return loss_second
```

**Helper methods**:
```python
def _set_requires_grad(self, model, layer_names, requires_grad):
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = requires_grad

def _freeze_layers(self, model, layer_names):
    self._set_requires_grad(model, layer_names, False)

def _activate_layers(self, model, layer_names):
    self._set_requires_grad(model, layer_names, True)
```

---

## Detailed Step-by-Step Comparison

### OLD Implementation Flow:

1. **First Pass (Local Layers)**:
   - `freeze_model(model, global_layer_names)` → Sets `requires_grad = False` for global layers
   - `activate_model(model, local_layer_names)` → Sets `requires_grad = True` for local layers
   - `super().perform_forward_and_backward_passes()`:
     - `optimizer.zero_grad()`
     - Forward pass: `outputs = model(examples)`
     - Compute loss: `loss = loss_criterion(outputs, labels)`
     - `loss.backward()` → Gradients computed only for local layers (global frozen)
     - `optimizer.step()` → Updates only local layer parameters

2. **Second Pass (Global Layers)**:
   - `activate_model(model, global_layer_names)` → Sets `requires_grad = True` for global layers
   - `freeze_model(model, local_layer_names)` → Sets `requires_grad = False` for local layers
   - `super().perform_forward_and_backward_passes()`:
     - `optimizer.zero_grad()`
     - Forward pass: `outputs = model(examples)`
     - Compute loss: `loss = loss_criterion(outputs, labels)`
     - `loss.backward()` → Gradients computed only for global layers (local frozen)
     - `optimizer.step()` → Updates only global layer parameters

3. **Return**: Loss from second pass (global layer training)

### NEW Implementation Flow (with train_local_first=True):

1. **First Pass (Local Layers)**:
   - `_freeze_layers(model, global_layer_names)` → Sets `requires_grad = False` for global layers
   - `_activate_layers(model, local_layer_names)` → Sets `requires_grad = True` for local layers
   - `optimizer.zero_grad()`
   - Forward pass: `outputs = model(examples)`
   - Compute loss: `loss_first = loss_criterion(outputs, labels)`
   - `loss_first.backward()` → Gradients computed only for local layers (global frozen)
   - `optimizer.step()` → Updates only local layer parameters

2. **Second Pass (Global Layers)**:
   - `_freeze_layers(model, local_layer_names)` → Sets `requires_grad = False` for local layers
   - `_activate_layers(model, global_layer_names)` → Sets `requires_grad = True` for global layers
   - `optimizer.zero_grad()`
   - Forward pass: `outputs = model(examples)`
   - Compute loss: `loss_second = loss_criterion(outputs, labels)`
   - `loss_second.backward()` → Gradients computed only for global layers (local frozen)
   - `optimizer.step()` → Updates only global layer parameters

3. **Cleanup**:
   - `_activate_layers(model, global_layer_names)` → Re-enable gradients for global layers
   - `_activate_layers(model, local_layer_names)` → Re-enable gradients for local layers

4. **Return**: Loss from second pass (global layer training)

---

## Comparison Analysis

### Similarities ✓
1. **Same algorithmic flow**: Both implementations perform two forward/backward passes per iteration
2. **Same training order**: Train local layers first, then global layers (when train_local_first=True)
3. **Same layer freezing mechanism**: Use `requires_grad` flag to freeze/activate layers
4. **Same pattern matching**: Both use substring matching (`layer_name in name`) to identify parameters
5. **Same loss return**: Both return the loss from the second pass (global layer training)
6. **Identical core operations**: 
   - Freeze global → Train local
   - Freeze local → Train global
   - Each pass includes: zero_grad → forward → loss → backward → step

### Differences

#### 1. **Architecture Pattern**
- **OLD**: Inheritance-based (extends `basic.Trainer`)
- **NEW**: Composition-based (uses `TrainingStepStrategy`)

#### 2. **Code Organization**
- **OLD**: Overrides a method in the trainer class hierarchy
- **NEW**: Encapsulates logic in a standalone strategy class

#### 3. **Flexibility**
- **OLD**: Fixed order (always local first, then global)
- **NEW**: Configurable order via `train_local_first` parameter

#### 4. **Gradient State Cleanup**
- **OLD**: Does NOT re-enable gradients after training (leaves local layers frozen)
- **NEW**: Re-enables gradients for both layer sets after training

#### 5. **Additional Features in NEW**
- `LGFedAvgStepStrategyFromConfig`: Reads config automatically
- `LGFedAvgStepStrategyAuto`: Auto-detects layer names
- Better documentation and type hints

---

## Verification Against Paper

Based on the paper "Think Locally, Act Globally: Federated Learning with Local and Global Representations" (Liang et al., NeurIPS 2019):

### Key Algorithm Requirements:

1. **✓ Model Partitioning**: Divide model into global (θ_g) and local (θ_l) parameters
   - Both implementations support this via layer name configuration

2. **✓ Dual Training Passes**: Perform two optimization steps per iteration
   - Both implementations perform two forward/backward passes
   - First pass updates local parameters only
   - Second pass updates global parameters only

3. **✓ Parameter Freezing**: During each pass, freeze one set while training the other
   - Both implementations use `requires_grad` to freeze/activate parameters

4. **✓ Server Aggregation**: Only global parameters are aggregated on server
   - Handled by the personalized FL infrastructure (fedavg_personalized)
   - Local parameters stay on clients

5. **Algorithm 1 from paper (Client Update)**:
   ```
   for each local epoch e:
       for each batch (x, y):
           # Update local parameters
           θ_l ← θ_l - η ∇_θl L(x, y; θ_g, θ_l)  
           
           # Update global parameters  
           θ_g ← θ_g - η ∇_θg L(x, y; θ_g, θ_l)
   ```
   
   **Both implementations match this exactly**:
   - First pass: Freeze θ_g, update θ_l via gradient descent
   - Second pass: Freeze θ_l, update θ_g via gradient descent

### Loss Tracking: Different but Correct

**OLD Implementation**: Loss tracking happens INSIDE the training step method:
```python
def perform_forward_and_backward_passes(self, config, examples, labels):
    self.optimizer.zero_grad()
    outputs = self.model(examples)
    loss = self._loss_criterion(outputs, labels)
    self._loss_tracker.update(loss, labels.size(0))  # ← Tracked here
    loss.backward()
    self.optimizer.step()
    return loss
```

**NEW Implementation**: Loss tracking happens OUTSIDE the strategy, in the composable trainer framework:
```python
# In ComposableTrainer.train_model() at line 345-348:
loss = self.training_step_strategy.training_step(
    model=self.model,
    optimizer=self.optimizer,
    examples=examples,
    labels=labels,
    loss_criterion=compute_loss,
    context=self.context,
)
# Track loss (outside strategy)
self._loss_tracker.update(loss, labels.size(0))  # ← Tracked here
```

This is a **better separation of concerns**: The strategy focuses on algorithm logic (how to train), while the framework handles metrics tracking (what to log). Both approaches track the same loss value from the second pass.

---

## Conclusion

### Algorithm Equivalence: **YES** ✓

Both implementations are **algorithmically identical** and correctly implement the LG-FedAvg algorithm from the paper:

1. Both perform two forward/backward passes per training step
2. Both freeze global layers when training local layers
3. Both freeze local layers when training global layers
4. Both use the same parameter matching logic
5. Both follow the algorithm description from Liang et al. (2019)

### Key Difference: Gradient State Management

The one behavioral difference is:
- **OLD**: Leaves local layers frozen after training step completes
- **NEW**: Re-enables all gradients after training step completes

The NEW approach is more correct as it ensures the model is in a clean state after each training step. However, this doesn't affect the algorithm correctness since the freezing/activation is reset at the start of each training step anyway.

### Paper Consistency: **YES** ✓

Both implementations correctly follow the LG-FedAvg algorithm as described in:
> Liang et al., "Think Locally, Act Globally: Federated Learning with Local and Global Representations," NeurIPS 2019

The implementations properly:
- Separate model into local and global parameters
- Perform alternating optimization of local and global parameters
- Use standard gradient descent for each optimization step
- Support the personalized FL framework where only global parameters are aggregated

The new composition-based implementation adds flexibility (training order, auto-detection) while maintaining algorithmic fidelity to the original paper.
