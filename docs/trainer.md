# Trainers

## Customizing trainers using inheritance

The common practice is to customize the training loop using inheritance for important features that change the state of the training process. To customize the training loop using inheritance, subclass the `basic.Trainer` class in `plato.trainers`, and override the following methods:

````{admonition} **get_train_loader(cls, batch_size, trainset, sampler, \*\*kwargs)**
This is a class method that is called to create an instance of the trainloader to be used in the training loop.

`batch_size` the batch size.

`trainset` the training dataset.

`sampler` the sampler for the trainloader to use.

```py
def get_train_loader(cls, batch_size, trainset, sampler, **kwargs):
    return torch.utils.data.DataLoader(
        dataset=trainset, shuffle=False, batch_size=batch_size, sampler=sampler
    )
```
````

```{admonition} **get_optimizer(self, model)**
Returns a custom optimizer.
```

```{admonition} **get_lr_scheduler(self, config, optimizer)**
Returns a custom learning rate scheduler.
```

```{admonition} **get_loss_criterion(self)**
Returns a custom loss criterion.
```

```{admonition} **lr_scheduler_step(self)**
Performs a single learning rate scheduler step if ``self.lr_scheduler`` has been assigned.
```

````{admonition} **train_run_start(self, config)**
Override this method to complete additional tasks before the training loop starts.

`config` the configuration settings used in the training loop. It corresponds directly to the `trainer` section in the configuration file.

**Example:**

```py
def train_run_start(self, config):
    logging.info("[Client #%d] Loading the dataset.", self.client_id)
```
````

````{admonition} **train_run_end(self, config)**
Override this method to complete additional tasks after the training loop ends.

`config` the configuration settings used in the training loop. It corresponds directly to the `trainer` section in the configuration file.

**Example:**

```py
def train_run_end(self, config):
    logging.info("[Client #%d] Completed the training loop.", self.client_id)
```
````

````{admonition} **train_epoch_start(self, config)**
Override this method to complete additional tasks at the starting point of each training epoch.

`config` the configuration settings used in the training loop. It corresponds directly to the `trainer` section in the configuration file.

**Example:**

```py
def train_epoch_start(self, config):
    logging.info("[Client #%d] Started training epoch %d.", self.client_id, self.current_epoch)
```
````

````{admonition} **train_epoch_end(self, config)**
Override this method to complete additional tasks at the end of each training epoch.

`config` the configuration settings used in the training loop. It corresponds directly to the `trainer` section in the configuration file.

**Example:**

```py
def train_epoch_end(self, config):
    logging.info("[Client #%d] Finished training epoch %d.", self.client_id, self.current_epoch)
```
````

````{admonition} **train_step_end(self, config, batch=None, loss=None)**
Override this method to complete additional tasks at the end of each step within a training epoch.

`config` the configuration settings used in the training loop. It corresponds directly to the `trainer` section in the configuration file.

`batch` the index of the current batch of data just processed in the currents step.

`loss` the loss value computed using the current batch of data after training.

**Example:**

```py
def train_epoch_end(self, config):
    logging.info(
        "[Client #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
        self.client_id,
        self.current_epoch,
        config["epochs"],
        batch,
        len(self.train_loader),
        loss.data.item(),
    )
````

## Customizing trainers using callbacks

For infrastructure changes, such as logging, recording metrics, and stopping the training loop early, we tend to customize the training loop using callbacks instead. The advantage of using callbacks is that one can pass a list of multiple callbacks to the trainer when it is initialized, and they will be called in their order in the provided list. This helps when it is necessary to group features into different callback classes.

Within the implementation of these callback methods, one can access additional information about the training loop by using the `trainer` instance. For example, `trainer.sampler` can be used to access the sampler used by the train dataloader, `trainer.trainloader` can be used to access the current train dataloader, and `trainer.current_epoch` can be used to access the current epoch number.

To use callbacks, subclass the `TrainerCallback` class in `plato.callbacks.trainer`, and override the following methods:

````{admonition} **on_train_run_start(self, trainer, config)**
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
````

````{admonition} **on_train_run_end(self, trainer, config)**
Override this method to complete additional tasks after the training loop ends.

`trainer` the trainer instance that activated this callback upon the occurrence of the corresponding event.

`config` the configuration settings used in the training loop. It corresponds directly to the `trainer` section in the configuration file.

**Example:**

```py
def on_train_run_end(self, trainer, config):
    logging.info("[Client #%d] Completed the training loop.", trainer.client_id)
```
````

````{admonition} **on_train_epoch_start(self, trainer, config)**
Override this method to complete additional tasks at the starting point of each training epoch.

`trainer` the trainer instance that activated this callback upon the occurrence of the corresponding event.

`config` the configuration settings used in the training loop. It corresponds directly to the `trainer` section in the configuration file.

**Example:**

```py
def train_epoch_start(self, trainer, config):
    logging.info("[Client #%d] Started training epoch %d.", trainer.client_id, trainer.current_epoch)
```
````

````{admonition} **on_train_epoch_end(self, trainer, config)**
Override this method to complete additional tasks at the end of each training epoch.

`trainer` the trainer instance that activated this callback upon the occurrence of the corresponding event.

`config` the configuration settings used in the training loop. It corresponds directly to the `trainer` section in the configuration file.

**Example:**

```py
def on_train_epoch_end(self, trainer, config):
    logging.info("[Client #%d] Finished training epoch %d.", trainer.client_id, trainer.current_epoch)
```
````

````{admonition} **on_train_step_end(self, trainer, config, batch=None, loss=None)**
Override this method to complete additional tasks at the end of each step within a training epoch.

`trainer` the trainer instance that activated this callback upon the occurrence of the corresponding event.

`config` the configuration settings used in the training loop. It corresponds directly to the `trainer` section in the configuration file.

`batch` the index of the current batch of data that has just been processed in the current step.

`loss` the loss value computed using the current batch of data after training.

**Example:**

```py
def on_train_epoch_end(self, trainer, config):
    logging.info(
        "[Client #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
        trainer.client_id,
        trainer.current_epoch,
        config["epochs"],
        batch,
        len(trainer.train_loader),
        loss.data.item(),
    )
````

## Accessing and customizing the run history during training

An instance of the `plato.trainers.tracking.RunHistory` class, called `self.run_history`, is used to store any number of performance metrics during the training process, one iterable list of values for each performance metric. By default, it stores the average loss values in each epoch.

The run history in the trainer can be accessed by the client as well, using `self.trainer.run_history`.  It can also be read, updated, or reset in the hooks or callback methods. For example, in the implementation of some algorithms such as Oort, a per-step loss value needs to be stored by calling `update_metric()` in `train_step_end()`:

```py
def train_step_end(self, config, batch=None, loss=None):
    self.run_history.update_metric("train_loss_step", loss.cpu().detach().numpy())
```

Here is a list of all the methods available in the `RunHistory` class:

```{admonition} **get_metric_names(self)**
Returns an iterable set containing of all unique metric names which are being tracked.
```

```{admonition} **get_metric_values(self, metric_name)**
Returns an ordered iterable list of values that has been stored since the last reset corresponding to the provided metric name.
```

```{admonition} **get_latest_metric(self, metric_name)**
Returns the most recent value that has been recorded for the given metric.
```

```{admonition} **update_metric(self, metric_name, metric_value)**
Records a new value for the given metric.
```

```{admonition} **reset(self)**
Resets the run history.
```