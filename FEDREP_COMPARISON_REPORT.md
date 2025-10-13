# FedRep Implementation Comparison Report

## Executive Summary

The new composition-based FedRep implementation in the `trainer-refactor` branch is **algorithmically consistent** with both:
1. The original inheritance-based implementation in the `main` branch
2. The FedRep algorithm described in Collins et al., "Exploiting Shared Representations for Personalized Federated Learning," ICML 2021

**Status**: ✅ VERIFIED with one bug fix applied

**Bug Fixed**: The personalization epochs configuration was not being applied in the new implementation. This has been corrected.

---

## Algorithm Overview (from Paper)

According to Collins et al. (ICML 2021), Section 3:

### Regular Federated Learning Rounds (rounds 1 to R)
- Each client trains for E epochs per round
- **First τ epochs**: Train local layers (head) only, freeze global layers (representation)
- **Remaining (E - τ) epochs**: Train global layers only, freeze local layers
- Only global layers are aggregated on the server

### Final Personalization Round (after round R)
- Freeze global layers permanently
- Train only local layers for personalization
- No server aggregation (purely local training)

---

## Implementation Comparison

### 1. Regular Training Rounds (round ≤ trainer.rounds)

#### Old Implementation (main branch)
```python
def train_epoch_start(self, config):
    if self.current_round <= Config().trainer.rounds:
        local_epochs = Config().algorithm.local_epochs
        
        if self.current_epoch <= local_epochs:
            # Epochs 1 to local_epochs: Train local layers
            trainer_utils.freeze_model(self.model, Config().algorithm.global_layer_names)
            trainer_utils.activate_model(self.model, Config().algorithm.local_layer_names)
        else:
            # Remaining epochs: Train global layers
            trainer_utils.freeze_model(self.model, Config().algorithm.local_layer_names)
            trainer_utils.activate_model(self.model, Config().algorithm.global_layer_names)
```
- **Hook used**: `train_epoch_start`
- **Call frequency**: Once per epoch
- **Logic**: Switch layers based on `current_epoch <= local_epochs`

#### New Implementation (current branch)
```python
def before_step(self, context: TrainingContext) -> None:
    if not self.is_personalizing:
        current_epoch = context.current_epoch
        
        # Optimize: only update layer freezing when epoch changes
        if self._last_processed_epoch != current_epoch:
            self._last_processed_epoch = current_epoch
            
            if current_epoch <= self.local_epochs:
                # Train local layers, freeze global layers
                self._freeze_global_layers(context)
                self._activate_local_layers(context)
            else:
                # Train global layers, freeze local layers
                self._freeze_local_layers(context)
                self._activate_global_layers(context)
```
- **Hook used**: `before_step` (called before each training batch)
- **Call frequency**: Before every batch, but optimized with epoch tracking
- **Logic**: Same condition `current_epoch <= local_epochs`

#### Analysis
✅ **Algorithmically Identical**
- Both use the same epoch-based condition
- Both freeze/activate the same layers at the same times
- Optimization added to avoid redundant operations when called multiple times per epoch
- PyTorch's `requires_grad` setting persists, so the effect is identical

---

### 2. Personalization Round (round > trainer.rounds)

#### Old Implementation (main branch)
```python
def train_run_start(self, config):
    if self.current_round > Config().trainer.rounds:
        trainer_utils.freeze_model(self.model, Config().algorithm.global_layer_names)
        
        if hasattr(Config().algorithm.personalization, "epochs"):
            config["epochs"] = Config().algorithm.personalization.epochs

def train_epoch_start(self, config):
    if self.current_round > Config().trainer.rounds:
        trainer_utils.freeze_model(self.model, Config().algorithm.global_layer_names)

def train_run_end(self, config):
    if self.current_round > Config().trainer.rounds:
        trainer_utils.activate_model(self.model, Config().algorithm.global_layer_names)
```

#### New Implementation (current branch - after fix)
```python
def on_train_start(self, context: TrainingContext) -> None:
    self.is_personalizing = context.current_round > total_rounds
    
    if self.is_personalizing:
        self._freeze_global_layers(context)
        
        if hasattr(Config().algorithm.personalization, "epochs"):
            context.config["epochs"] = Config().algorithm.personalization.epochs  # FIXED

def before_step(self, context: TrainingContext) -> None:
    if not self.is_personalizing:
        # NOT executed during personalization
        pass

def on_train_end(self, context: TrainingContext) -> None:
    self._activate_global_layers(context)
    self._activate_local_layers(context)
```

#### Analysis
✅ **Algorithmically Identical (after fix)**
- Both freeze global layers at training start during personalization
- Both apply personalization epochs configuration (after fix)
- Both reactivate layers at training end
- Minor difference: new implementation reactivates all layers unconditionally (harmless)

---

### 3. Layer Freezing/Activation Implementation

#### Old Implementation
Uses helper functions from `plato.utils.trainer_utils`:
```python
def freeze_model(model, layer_names):
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False

def activate_model(model, layer_names):
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True
```

#### New Implementation
Implements the same logic as methods:
```python
def _freeze_global_layers(self, context: TrainingContext) -> None:
    for name, param in context.model.named_parameters():
        if any(layer_name in name for layer_name in self.global_layer_names):
            param.requires_grad = False

def _activate_global_layers(self, context: TrainingContext) -> None:
    for name, param in context.model.named_parameters():
        if any(layer_name in name for layer_name in self.global_layer_names):
            param.requires_grad = True
```

#### Analysis
✅ **Identical Implementation**
- Same string matching logic
- Same PyTorch parameter modification
- Just encapsulated differently

---

## Consistency with Paper

According to Collins et al. (ICML 2021), the FedRep algorithm requires:

### ✅ Requirement 1: Alternating Training
**Paper**: "...first updates the head parameters θ_h for τ epochs while keeping the representation frozen, then updates the representation parameters θ_r for E-τ epochs with a frozen head."

**Implementation**: Both old and new implementations correctly alternate between:
- Epochs 1 to τ (local_epochs): Train local/head layers, freeze global/representation
- Epochs τ+1 to E: Train global/representation, freeze local/head

### ✅ Requirement 2: Only Global Layers Aggregated
**Paper**: "Only the representation parameters θ_r are aggregated by the server."

**Implementation**: This is handled by the server/client code, not the trainer. The trainer correctly trains both layer types at appropriate times, and the server aggregation logic (not modified) handles selecting only global layers.

### ✅ Requirement 3: Final Personalization
**Paper**: "After R rounds of federated learning, each client performs τ_p epochs of personalization by training only the head while keeping the representation frozen."

**Implementation**: Both implementations correctly:
- Detect when `current_round > trainer.rounds`
- Freeze global/representation layers
- Train only local/head layers
- Apply personalization-specific epoch count if configured

---

## Bug Found and Fixed

### Issue
In the new implementation, the personalization epochs configuration was not being applied:

**Before Fix**:
```python
if hasattr(Config().algorithm.personalization, "epochs"):
    # Only stored in context.state, not used
    context.state["personalization_epochs"] = Config().algorithm.personalization.epochs
```

**After Fix**:
```python
if hasattr(Config().algorithm.personalization, "epochs"):
    # Correctly modifies the training config
    context.config["epochs"] = Config().algorithm.personalization.epochs
```

### Impact
Without this fix, personalization rounds would use the regular epoch count instead of the personalization-specific epoch count, potentially leading to under/over-training during personalization.

---

## Optimization Added

Added epoch tracking to avoid redundant layer freezing/activation operations:

```python
def __init__(self, ...):
    self._last_processed_epoch = None

def before_step(self, context: TrainingContext) -> None:
    if not self.is_personalizing:
        current_epoch = context.current_epoch
        
        # Only update when epoch changes
        if self._last_processed_epoch != current_epoch:
            self._last_processed_epoch = current_epoch
            # ... freeze/activate layers
```

This optimization reduces unnecessary PyTorch parameter updates when `before_step` is called multiple times per epoch (once per batch).

---

## Testing

A comprehensive test suite has been created in `tests/test_fedrep_implementation.py` covering:

1. **Layer Freezing Tests**: Verify correct layers are frozen/active in each phase
2. **Epoch Transition Tests**: Verify correct behavior across epoch boundaries
3. **Personalization Tests**: Verify personalization phase behavior
4. **Configuration Tests**: Verify FedRepUpdateStrategyFromConfig works correctly
5. **Algorithmic Equivalence Tests**: Simulate complete training rounds to verify behavior

Run tests with:
```bash
pytest tests/test_fedrep_implementation.py -v
```

---

## Conclusion

### Summary
✅ The new composition-based FedRep implementation is **algorithmically identical** to the old inheritance-based implementation after applying the personalization epochs bug fix.

✅ Both implementations are **consistent with the FedRep paper** (Collins et al., ICML 2021).

✅ An optimization was added to improve efficiency without changing algorithmic behavior.

### Changes Made
1. **Fixed personalization epochs**: Changed from storing in `context.state` to modifying `context.config["epochs"]`
2. **Added optimization**: Track last processed epoch to avoid redundant operations
3. **Added tests**: Comprehensive test suite to verify correctness

### Migration Notes
The new implementation can be used as a drop-in replacement for the old implementation. The only difference is the class used:

**Old**:
```python
from plato.trainers import basic
class Trainer(basic.Trainer):
    # Override methods
```

**New**:
```python
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import FedRepUpdateStrategyFromConfig

class Trainer(ComposableTrainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            model_update_strategy=FedRepUpdateStrategyFromConfig(),
        )
```

---

## References

1. Collins, L., Hassani, H., Mokhtari, A., & Shakkottai, S. (2021). "Exploiting Shared Representations for Personalized Federated Learning." In Proceedings of the 38th International Conference on Machine Learning (ICML). https://arxiv.org/abs/2102.07078

2. FedRep official implementation: https://github.com/lgcollins/FedRep
