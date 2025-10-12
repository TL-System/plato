# Trainer Refactoring: Architecture Diagrams

This document provides visual representations of the current and proposed trainer architectures.

---

## 1. Current Architecture (Inheritance-Based)

```
┌─────────────────────────────────────────────────────────────┐
│                      base.Trainer                           │
│                    (Abstract Base)                          │
│  - save_model()                                             │
│  - load_model()                                             │
│  - train() [abstract]                                       │
│  - test() [abstract]                                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ inherits
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    basic.Trainer                            │
│                 (Concrete Implementation)                   │
│  + train_model(config, trainset, sampler)                  │
│  + get_loss_criterion()                                     │
│  + get_optimizer(model)                                     │
│  + get_lr_scheduler(config, optimizer)                      │
│  + perform_forward_and_backward_passes(...)                 │
│  + train_run_start(config)                                  │
│  + train_epoch_start(config)                                │
│  + train_step_start(config, batch)                          │
│  + train_step_end(config, batch, loss)                      │
│  + train_epoch_end(config)                                  │
│  + train_run_end(config)                                    │
│  ... (600+ lines)                                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┬──────────────────┐
        │             │             │                  │
        │ inherits    │ inherits    │ inherits         │ inherits
        ▼             ▼             ▼                  ▼
┌───────────────┐ ┌──────────┐ ┌──────────┐  ... ┌──────────┐
│ FedProx       │ │ SCAFFOLD │ │ FedDyn   │      │ LGFedAvg │
│ Trainer       │ │ Trainer  │ │ Trainer  │      │ Trainer  │
│               │ │          │ │          │      │          │
│ override:     │ │ override:│ │ override:│      │ override:│
│ - get_loss_   │ │ - __init │ │ - __init │      │ - perform│
│   criterion() │ │ - train_ │ │ - perform│      │   _forward│
│               │ │   run_   │ │   _forward│     │   ...    │
│               │ │   start()│ │   ...    │      │          │
│               │ │ - train_ │ │          │      │          │
│               │ │   step_  │ │          │      │          │
│               │ │   end()  │ │          │      │          │
│               │ │ - train_ │ │          │      │          │
│               │ │   run_   │ │          │      │          │
│               │ │   end()  │ │          │      │          │
└───────────────┘ └──────────┘ └──────────┘      └──────────┘

Problems:
✗ Cannot combine FedProx + SCAFFOLD without creating new subclass
✗ Tight coupling to base class implementation
✗ Difficult to test individual components
✗ Changes to base class affect all subclasses
✗ Code duplication across similar trainers
```

---

## 2. Proposed Architecture (Composition-Based)

```
┌──────────────────────────────────────────────────────────────┐
│                   ComposableTrainer                          │
│                                                              │
│  Constructor:                                                │
│  - loss_strategy: LossCriterionStrategy                     │
│  - optimizer_strategy: OptimizerStrategy                    │
│  - training_step_strategy: TrainingStepStrategy             │
│  - lr_scheduler_strategy: LRSchedulerStrategy               │
│  - model_update_strategy: ModelUpdateStrategy               │
│  - data_loader_strategy: DataLoaderStrategy                 │
│                                                              │
│  Training Loop:                                              │
│  + train_model(config, trainset, sampler):                  │
│      1. data_loader_strategy.create_train_loader()          │
│      2. optimizer_strategy.create_optimizer()               │
│      3. lr_scheduler_strategy.create_scheduler()            │
│      4. for epoch in epochs:                                │
│           model_update_strategy.on_train_start()            │
│           for batch in train_loader:                        │
│               training_step_strategy.training_step()        │
│               ├─ loss_strategy.compute_loss()               │
│               └─ model_update_strategy.after_step()         │
│           lr_scheduler_strategy.step()                      │
│      5. model_update_strategy.on_train_end()                │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         │ delegates to
                         ▼
        ┌────────────────────────────────────────┐
        │         Strategy Interfaces            │
        └────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Loss         │  │ Optimizer    │  │ Training     │
│ Criterion    │  │ Strategy     │  │ Step         │
│ Strategy     │  │              │  │ Strategy     │
│              │  │              │  │              │
│ - setup()    │  │ - setup()    │  │ - setup()    │
│ - compute_   │  │ - create_    │  │ - training_  │
│   loss()     │  │   optimizer()│  │   step()     │
│ - teardown() │  │ - teardown() │  │ - teardown() │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       │ implementations │                 │
       ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ FedProx      │  │ Adam         │  │ Default      │
│ Loss         │  │ Optimizer    │  │ Training     │
│ Strategy     │  │ Strategy     │  │ Step         │
│              │  │              │  │ Strategy     │
│ + mu param   │  │ + lr param   │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ FedDyn       │  │ SGD          │  │ LGFedAvg     │
│ Loss         │  │ Optimizer    │  │ Training     │
│ Strategy     │  │ Strategy     │  │ Step         │
│              │  │              │  │ Strategy     │
│ + alpha param│  │              │  │              │
└──────────────┘  └──────────────┘  └──────────────┘

        ┌────────────────┬────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ LR Scheduler │  │ Model Update │  │ Data Loader  │
│ Strategy     │  │ Strategy     │  │ Strategy     │
│              │  │              │  │              │
│ - create_    │  │ - on_train_  │  │ - create_    │
│   scheduler()│  │   start()    │  │   train_     │
│ - step()     │  │ - after_step()│ │   loader()   │
└──────┬───────┘  └──────┬───────┘  └──────────────┘
       │                 │
       │                 │ implementations
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│ Default      │  │ SCAFFOLD     │
│ LR           │  │ Update       │
│ Scheduler    │  │ Strategy     │
└──────────────┘  │              │
                  │ + control_   │
                  │   variates   │
                  └──────────────┘
                  ┌──────────────┐
                  │ FedDyn       │
                  │ Update       │
                  │ Strategy     │
                  └──────────────┘

Benefits:
✓ Easy composition: FedProx + SCAFFOLD = inject both strategies
✓ Loose coupling: Strategies independent of trainer
✓ Easy testing: Unit test each strategy separately
✓ Safe refactoring: Changes to one strategy don't affect others
✓ No code duplication: Strategies are reusable
```

---

## 3. Strategy Pattern Detail

```
┌─────────────────────────────────────────────────────────────┐
│                   Strategy Pattern                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 Abstract Strategy                           │
│                   (Interface)                               │
│                                                             │
│  + setup(context: TrainingContext) -> None                 │
│  + execute(...) -> Result              [abstract]          │
│  + teardown(context: TrainingContext) -> None              │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │Concrete │ │Concrete │ │Concrete │
    │Strategy │ │Strategy │ │Strategy │
    │   A     │ │   B     │ │   C     │
    └─────────┘ └─────────┘ └─────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Context                                 │
│  (Trainer holds reference to strategy and delegates)        │
│                                                             │
│  - strategy: AbstractStrategy                               │
│                                                             │
│  + execute_strategy(...):                                   │
│      return strategy.execute(...)                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. TrainingContext Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                  TrainingContext                            │
│  (Shared state container passed between strategies)         │
│                                                             │
│  Attributes:                                                │
│  • model: nn.Module                                         │
│  • device: torch.device                                     │
│  • client_id: int                                           │
│  • current_epoch: int                                       │
│  • current_round: int                                       │
│  • config: Dict[str, Any]                                   │
│  • state: Dict[str, Any]  <- Strategies can share data here │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ passed to
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Strategy A   │    │ Strategy B   │    │ Strategy C   │
│              │    │              │    │              │
│ Reads:       │    │ Reads:       │    │ Reads:       │
│ - model      │    │ - model      │    │ - state['X'] │
│ - device     │    │ - config     │    │ - device     │
│              │    │              │    │              │
│ Writes:      │    │ Writes:      │    │ Writes:      │
│ - state['X'] │───┼─▶ [reads X]   │    │ - state['Y'] │
│              │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘

Example: SCAFFOLD shares control variates via context.state
```

---

## 5. Training Loop Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│             ComposableTrainer.train_model()                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ 1. Setup Phase                        │
        │  • Create data loader (strategy)      │
        │  • Create optimizer (strategy)        │
        │  • Create LR scheduler (strategy)     │
        │  • model_update_strategy.on_train_    │
        │    start()                            │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ 2. Training Loop                      │
        │  FOR epoch IN epochs:                 │
        │    ├─ callbacks.on_train_epoch_start()│
        │    │                                   │
        │    FOR batch IN train_loader:         │
        │      ├─ callbacks.on_train_step_start()│
        │      │                                 │
        │      ├─ model_update_strategy.        │
        │      │   before_step()                │
        │      │                                 │
        │      ├─ training_step_strategy.       │
        │      │   training_step():             │
        │      │   ├─ forward pass              │
        │      │   ├─ loss_strategy.compute_    │
        │      │   │   loss()                   │
        │      │   ├─ backward pass             │
        │      │   └─ optimizer.step()          │
        │      │                                 │
        │      ├─ model_update_strategy.        │
        │      │   after_step()                 │
        │      │                                 │
        │      └─ callbacks.on_train_step_end() │
        │    │                                   │
        │    ├─ lr_scheduler_strategy.step()    │
        │    └─ callbacks.on_train_epoch_end()  │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ 3. Cleanup Phase                      │
        │  • model_update_strategy.on_train_    │
        │    end()                              │
        │  • callbacks.on_train_run_end()       │
        │  • Return results                     │
        └───────────────────────────────────────┘
```

---

## 6. Strategy Composition Example: FedProx + SCAFFOLD

```
┌─────────────────────────────────────────────────────────────┐
│           ComposableTrainer with Multiple Strategies        │
│                                                             │
│  trainer = ComposableTrainer(                               │
│      loss_strategy=FedProxLossStrategy(mu=0.01),           │
│      model_update_strategy=SCAFFOLDUpdateStrategy()        │
│  )                                                          │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│  FedProxLossStrategy     │    │ SCAFFOLDUpdateStrategy   │
│                          │    │                          │
│  Handles:                │    │ Handles:                 │
│  • Proximal term in loss │    │ • Control variates       │
│  • ||w - w_global||²     │    │ • Gradient correction    │
│                          │    │ • State management       │
│  Called during:          │    │                          │
│  • loss computation      │    │ Called during:           │
│                          │    │ • train_start()          │
│  Dependencies:           │    │ • after_step()           │
│  • Needs global weights  │    │ • train_end()            │
│  • Saved at setup()      │    │                          │
└──────────────────────────┘    │ Dependencies:            │
                                │ • Server control variate │
                                │ • Client control variate │
                                │ • Learning rate          │
                                └──────────────────────────┘

Execution Flow:
─────────────────────────────────────────────────────────────
Setup:
  1. FedProx: Save global model weights
  2. SCAFFOLD: Load/initialize control variates

Training Step:
  1. SCAFFOLD: Apply control variate correction (before)
  2. Forward pass
  3. FedProx: Compute loss with proximal term
  4. Backward pass
  5. Optimizer step
  6. SCAFFOLD: Apply control variate correction (after)

Cleanup:
  1. SCAFFOLD: Compute new control variate
  2. FedProx: Clear saved weights

Result: Combined FedProx + SCAFFOLD algorithm without code duplication!
```

---

## 7. Class Diagram: Strategy Interfaces

```
┌─────────────────────────────────────────────────────────────┐
│                       <<interface>>                         │
│                        Strategy                             │
├─────────────────────────────────────────────────────────────┤
│ + setup(context: TrainingContext): void                     │
│ + teardown(context: TrainingContext): void                  │
└─────────────────────────────────────────────────────────────┘
                            △
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        │                   │                   │
┌───────┴──────────┐ ┌──────┴─────────┐ ┌──────┴──────────┐
│ <<interface>>    │ │ <<interface>>  │ │ <<interface>>   │
│ LossCriterion    │ │ Optimizer      │ │ TrainingStep    │
│ Strategy         │ │ Strategy       │ │ Strategy        │
├──────────────────┤ ├────────────────┤ ├─────────────────┤
│ + compute_loss() │ │ + create_      │ │ + training_     │
│   (outputs,      │ │   optimizer()  │ │   step(...)     │
│    labels,       │ │                │ │                 │
│    context)      │ │                │ │                 │
│   -> Tensor      │ │                │ │                 │
└───────┬──────────┘ └────────┬───────┘ └────────┬────────┘
        │                     │                  │
        │                     │                  │
┌───────┴──────────┐  ┌───────┴────────┐  ┌─────┴────────┐
│ FedProxLoss      │  │ AdamOptimizer  │  │ DefaultStep  │
│ Strategy         │  │ Strategy       │  │ Strategy     │
├──────────────────┤  ├────────────────┤  ├──────────────┤
│ - mu: float      │  │ - lr: float    │  │              │
│ - global_weights │  │ - betas        │  │              │
├──────────────────┤  ├────────────────┤  ├──────────────┤
│ + compute_loss() │  │ + create_      │  │ + training_  │
│                  │  │   optimizer()  │  │   step()     │
└──────────────────┘  └────────────────┘  └──────────────┘
┌──────────────────┐  ┌────────────────┐  ┌──────────────┐
│ FedDynLoss       │  │ SGDOptimizer   │  │ LGFedAvgStep │
│ Strategy         │  │ Strategy       │  │ Strategy     │
├──────────────────┤  ├────────────────┤  ├──────────────┤
│ - alpha: float   │  │ - lr: float    │  │ - global_    │
│ - local_params   │  │ - momentum     │  │   layers     │
├──────────────────┤  ├────────────────┤  │ - local_     │
│ + compute_loss() │  │ + create_      │  │   layers     │
│                  │  │   optimizer()  │  ├──────────────┤
└──────────────────┘  └────────────────┘  │ + training_  │
                                          │   step()     │
                                          └──────────────┘

        ┌───────────────────┬───────────────────┐
        │                   │                   │
┌───────┴──────────┐ ┌──────┴─────────┐ ┌──────┴──────────┐
│ <<interface>>    │ │ <<interface>>  │ │ <<interface>>   │
│ LRScheduler      │ │ ModelUpdate    │ │ DataLoader      │
│ Strategy         │ │ Strategy       │ │ Strategy        │
├──────────────────┤ ├────────────────┤ ├─────────────────┤
│ + create_        │ │ + on_train_    │ │ + create_train_ │
│   scheduler()    │ │   start()      │ │   loader()      │
│ + step()         │ │ + before_step()│ │                 │
│                  │ │ + after_step() │ │                 │
│                  │ │ + on_train_    │ │                 │
│                  │ │   end()        │ │                 │
│                  │ │ + get_update_  │ │                 │
│                  │ │   payload()    │ │                 │
└───────┬──────────┘ └────────┬───────┘ └────────┬────────┘
        │                     │                  │
┌───────┴──────────┐  ┌───────┴────────┐  ┌─────┴────────┐
│ DefaultLR        │  │ SCAFFOLDUpdate │  │ DefaultData  │
│ Scheduler        │  │ Strategy       │  │ Loader       │
│ Strategy         │  ├────────────────┤  │ Strategy     │
└──────────────────┘  │ - server_cv    │  └──────────────┘
                      │ - client_cv    │
                      │ - global_      │
                      │   weights      │
                      ├────────────────┤
                      │ + on_train_    │
                      │   start()      │
                      │ + after_step() │
                      │ + on_train_    │
                      │   end()        │
                      └────────────────┘
```

---

## 8. Migration Path Visualization

```
┌─────────────────────────────────────────────────────────────┐
│                  Migration Stages                           │
└─────────────────────────────────────────────────────────────┘

Stage 0: Current (Inheritance)
─────────────────────────────────────────────────────────────
Code:
  class MyTrainer(basic.Trainer):
      def get_loss_criterion(self):
          return custom_loss()

Status: ✓ Works  │  Recommended: No

─────────────────────────────────────────────────────────────
Stage 1: Backward Compatible (Hybrid)
─────────────────────────────────────────────────────────────
Code:
  class MyTrainer(basic.Trainer):
      def __init__(self):
          super().__init__(
              loss_strategy=MyLossStrategy()
          )

Status: ✓ Works  │  Recommended: For migration

─────────────────────────────────────────────────────────────
Stage 2: Pure Composition (Recommended)
─────────────────────────────────────────────────────────────
Code:
  trainer = ComposableTrainer(
      loss_strategy=MyLossStrategy(),
      model_update_strategy=MyUpdateStrategy()
  )

Status: ✓ Works  │  Recommended: Yes

─────────────────────────────────────────────────────────────
Timeline:
─────────────────────────────────────────────────────────────
Now          +3 mo        +6 mo        +9 mo        +12 mo
 │             │            │            │             │
 │ v1.0        │ v1.5       │ v2.0       │ v2.5        │ v3.0
 │             │            │            │             │
 ├─────────────┼────────────┼────────────┼─────────────┤
 │             │            │            │             │
 │ Stage 0,1,2 │ Stage 1,2  │ Stage 2    │ Stage 2     │ Stage 2
 │ all work    │ warn Stage │ strong     │ final       │ only
 │             │ 0          │ warnings   │ warning     │
 │             │            │            │             │
 └─────────────┴────────────┴────────────┴─────────────┘
   Introduce    Soft         Strong       Final         Remove
   strategies   deprecation  deprecation  warning       legacy
```

---

## 9. Dependency Injection Patterns

```
┌─────────────────────────────────────────────────────────────┐
│              Pattern 1: Constructor Injection                │
└─────────────────────────────────────────────────────────────┘

trainer = ComposableTrainer(
    loss_strategy=FedProxLossStrategy(mu=0.01),
    optimizer_strategy=AdamOptimizerStrategy(lr=0.001)
)

Benefits:
✓ Dependencies explicit at creation time
✓ Immutable after construction
✓ Easy to test (pass mocks)

─────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────┐
│              Pattern 2: Setter Injection                     │
└─────────────────────────────────────────────────────────────┘

trainer = ComposableTrainer()
trainer.set_loss_strategy(FedProxLossStrategy(mu=0.01))
trainer.set_optimizer_strategy(AdamOptimizerStrategy(lr=0.001))

Benefits:
✓ Can change strategies after creation
✓ Optional dependencies

Use case: Runtime strategy swapping

─────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────┐
│              Pattern 3: Factory Injection                    │
└─────────────────────────────────────────────────────────────┘

trainer = TrainerFactory.create_fedprox_trainer(mu=0.01)

# Or with builder pattern:
trainer = (TrainerBuilder()
    .with_loss_strategy(FedProxLossStrategy(mu=0.01))
    .with_optimizer_strategy(AdamOptimizerStrategy(lr=0.001))
    .build()
)

Benefits:
✓ Hide complexity of strategy selection
✓ Provide sensible defaults
✓ Fluent API

Use case: Common configurations
```

---

## 10. Testing Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Testing Pyramid                            │
└─────────────────────────────────────────────────────────────┘

                    ▲
                   ╱ ╲
                  ╱   ╲
                 ╱ E2E ╲          10% - Full training runs
                ╱───────╲         • Real models, real data
               ╱         ╲        • Slow, comprehensive
              ╱Integration╲       • Validate algorithm correctness
             ╱─────────────╲
            ╱               ╲     30% - Strategy + Trainer
           ╱   Integration   ╲    • Real trainer, real strategies
          ╱───────────────────╲   • Medium speed
         ╱                     ╲  • Validate interactions
        ╱        Unit           ╲
       ╱────────────────────────╲ 60% - Individual strategies
      ╱           Tests           ╲ • Mock dependencies
     ╱_____________________________╲ • Fast, isolated
                                   • Validate logic

─────────────────────────────────────────────────────────────

Unit Test Example:
─────────────────────────────────────────────────────────────
Test: FedProxLossStrategy.compute_loss()
Given: Model with known weights
When: Compute loss with mu=0.01
Then: Loss = base_loss + (0.01/2) * ||w-w_global||²

Mock: TrainingContext, Model
Speed: <100ms
Coverage: 95%+

─────────────────────────────────────────────────────────────

Integration Test Example:
─────────────────────────────────────────────────────────────
Test: ComposableTrainer with FedProxLossStrategy
Given: Simple dataset, simple model
When: Train for 2 epochs
Then: Loss decreases, weights updated

Mock: None (real components)
Speed: ~5 seconds
Coverage: 85%+

─────────────────────────────────────────────────────────────

E2E Test Example:
─────────────────────────────────────────────────────────────
Test: FedProx on MNIST
Given: MNIST dataset, CNN model
When: Train for 5 epochs
Then: Accuracy > 85%

Mock: None
Speed: ~5 minutes
Coverage: Major algorithms
```

---

## 11. Performance Comparison

```
┌─────────────────────────────────────────────────────────────┐
│          Inheritance vs Composition Performance             │
└─────────────────────────────────────────────────────────────┘

Metric: Training Time (seconds per epoch)
─────────────────────────────────────────────────────────────

Inheritance:  ████████████████████████████ 28.5s
Composition: