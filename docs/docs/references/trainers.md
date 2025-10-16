# Trainers

## Strategy-Based Trainer Architecture

Plato trainers use the same composition model as clients and servers. Every
`ComposableTrainer` instance wires a small set of interchangeable strategies,
letting you swap behaviour without subclassing:

- `LossCriterionStrategy` computes the objective.
- `OptimizerStrategy` builds and updates the optimiser.
- `TrainingStepStrategy` runs the forward/backward pass.
- `LRSchedulerStrategy` adjusts learning rates.
- `ModelUpdateStrategy` maintains auxiliary state (control variates, fine-tuning).
- `DataLoaderStrategy` creates train/test loaders.
- `TestingStrategy` evaluates the model.

Strategies share state through `TrainingContext`, which mirrors the trainer’s
model, optimiser, device, round counters, and an extensible `state` dictionary.

## Quick Start

```py
from plato.trainers.composable import ComposableTrainer

# Default stack: sensible strategies for supervised learning.
trainer = ComposableTrainer(model=my_model_fn)

# Mix and match to customise behaviour.
from plato.trainers.strategies import AdamOptimizerStrategy
from plato.trainers.strategies.algorithms import FedProxLossStrategy

fedprox_trainer = ComposableTrainer(
    model=my_model_fn,
    loss_strategy=FedProxLossStrategy(mu=0.01),
    optimizer_strategy=AdamOptimizerStrategy(lr=1e-3),
)
```

Pass `trainer=fedprox_trainer` when instantiating clients or servers to reuse the
same strategy stack in every round.

## Strategy Extension Points

- **`LossCriterionStrategy`**: add regularisers or alternate objectives; pull round
  metadata from `context` when needed.
- **`OptimizerStrategy`**: build custom optimisers or parameter groups; return a
  ready-to-use optimiser instance.
- **`TrainingStepStrategy`**: implement bespoke loops (LG-FedAvg, gradient clipping);
  keep tensors on device and reuse the supplied `loss_criterion`.
- **`LRSchedulerStrategy`**: wire warmup or timm schedulers by overriding
  `create_scheduler` and optional lifecycle hooks.
- **`ModelUpdateStrategy`**: persist control variates or personalised heads in
  `context.state`.
- **`DataLoaderStrategy`**: control sampling, augmentation, or worker config while
  honouring batch sizes from the config.
- **`TestingStrategy`**: customise evaluation logic and return scalar metrics for
  downstream logging.

Each concrete strategy inherits optional `setup`/`teardown` hooks and can emit
callback events via `context.callback_handler`.

## Composing Trainers

`ComposableTrainer` accepts either concrete strategy instances or `None` for the
defaults. You can start from `plato.trainers.basic.Trainer` (which simply wraps
the defaults) and override only the pieces you need:

```py
from plato.trainers.basic import Trainer
from plato.trainers.strategies.training_step import GradientClipStepStrategy

class ClippedTrainer(Trainer):
    def __init__(self, *, model=None, callbacks=None, max_norm=1.0):
        super().__init__(model=model, callbacks=callbacks)
        self._configure_composable(
            loss_strategy=self.loss_strategy,
            optimizer_strategy=self.optimizer_strategy,
            training_step_strategy=GradientClipStepStrategy(max_norm=max_norm),
            lr_scheduler_strategy=self.lr_scheduler_strategy,
            model_update_strategy=self.model_update_strategy,
            data_loader_strategy=self.data_loader_strategy,
            testing_strategy=self.testing_strategy,
        )
```

Strategies can also be registered in experiment configs—see the references under
`plato.trainers.strategies` for ready-made options such as FedNova, Scaffold,
and adaptation methods.

## Trainer Context and Run History

`TrainingContext` exposes:

- `model`, `optimizer`, `lr_scheduler`, and active data loaders.
- `client_id`, `current_round`, `current_epoch`, and `device`.
- `state` and `metadata` dictionaries for cross-strategy coordination.
- `run_history`, which records loss and accuracy per epoch/round.

Use these fields instead of storing state on the trainer subclass directly.

## Callbacks

Callbacks remain the recommended way to add logging or metrics. Subclass
`plato.callbacks.trainer.TrainerCallback`, override hooks such as
`on_train_epoch_start` or `on_test_end`, and pass the callback class through the
trainer constructor. Strategies can reuse the same callback pipeline by calling
`context.callback_handler.call_event(...)`.

## Legacy Hooks

Historic trainers that inherited `basic.Trainer` and overrode methods like
`train`, `test`, or `_load_weights` continue to work. The backward-compatible
constructor still installs the default strategies and forwards legacy hooks
through them. When migrating, move custom logic into dedicated strategies and
remove the override once behaviour matches the new pipeline.
