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

`ComposableTrainer` accepts either concrete strategy instances or `None` for the defaults. You can start from `plato.trainers.basic.Trainer` (which simply wraps the defaults) and override only the pieces you need:

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

## Example: Creating a Custom Strategy

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

To use the custom strategy:

```python
from plato.trainers.composable import ComposableTrainer

trainer = ComposableTrainer(
    model=my_model,
    loss_strategy=MyCustomLossStrategy(weight=0.01)
)
```

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

!!! note "`train_model()`"
    **`def train_model(self, config, trainset, sampler, **kwargs):`**

    Override this method to provide a custom training loop.

    `config` A dictionary of configuration parameters.
    `trainset` The training dataset.
    `sampler` the sampler that extracts a partition for this client.

    **Example:** A complete example can be found in the Hugging Face trainer, located at `plato/trainers/huggingface.py`.

!!! note "`test_model()`"
    **`def test_model(self, config, testset, sampler=None, **kwargs):`**

    Override this method to provide a custom testing loop.

    `config` A dictionary of configuration parameters.
    `testset` The test dataset.

    **Example:** A complete example can be found in `plato/trainers/huggingface.py`.

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

---
