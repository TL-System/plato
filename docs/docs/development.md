
# Developer's Guide

The Plato framework is designed to be extensible, hopefully making it easy to add new data sources for datasets, models, and custom trainers for models. This document discusses the current design of the framework from a software engineering perspective.

This framework makes extensive use of object oriented subclassing with the help of Python 3's [ABC library](https://docs.python.org/3/library/abc.html). It is a good idea to review Python 3's support for base classes with abstract methods before proceeding. It also makes sporadic use of Python 3's [Data Classes](https://docs.python.org/3/library/dataclasses.html). It also supports defining callback classes, and customizing a trainer by providing it with a list of custom callback classes.

---

## Configuration Parameters

All configuration parameters are globally accessed using the Singleton `Config` class (found in `config.py`). They are read from a configuration file when the clients and the servers launch, and the configuration file follows the TOML format for the sake of simplicity and readability.

These parameters include parameters specific to:

- Dataset configuration
- Data distribution settings
- Trainer configuration
- Federated learning algorithm settings
- Server configuration
- Cross-silo training parameters

Either a command-line argument (`-c` or `--config`) or an environment variable `config_file` can be used to specify the location of the configuration file.

Use `Config()` anywhere in the framework to access these configuration parameters.

---

## Extensible Modules

This framework breaks commonly shared components in a federated learning training workload into extensible modules that are as independent as possible.

### Data Sources

A `Datasource` instance is used to obtain the dataset, labels, and any data augmentation. For example, the `DataLoader` class in `torch.utils.data` can be used to load the MNIST dataset; `Datasets` classes in the `HuggingFace` framework can also be used as a data source to load datasets.

A data source must subclass the `Datasource` abstract base classes in `datasources/base.py`. This class may use third-party frameworks to load datasets, and may add additional functionality to support build-in transformations.

The external interface of this module is contained in `datasources/registry.py`. The registry contains a list of provided datasources in the framework, so that they can be discovered and loaded. Its most important function is `get()`, which returns a `DataSource` instance.

### Samplers

A `Sampler` is responsible for sampling a dataset for local training or testing at each client in the federated learning workload. This is used to *simulate* a local dataset that is available locally at the client, using either an i.i.d. or non-i.i.d. distribution.

For non-i.i.d. distributions, an example sampler that is based on the Dirichlet distribution (with a configurable concentration bias) is provided. Samplers are passed as one of the parameters to a `Dataloader` instance.

### Models

Plato directly uses models from PyTorch. The model registry (`models/registry.py`) returns a suitable model based on the model type and model name supplied from the configuration file.

The model type specifies the repository from which the model should be retrieved, such as:

- [PyTorch Hub](https://pytorch.org/hub/)
- [HuggingFace](https://huggingface.co/)

The model name is used to retrieve the corresponding model from the repository.

If the model type is not supplied by the configuration file, the model name is used to retrieve one of the basic models provided by Plato for benchmarking purposes. In addition to using the registry, a custom model class can be directly passed into the client and server for them to instantiate a model instance when needed.

---

## Extending Plato with New Federated Learning Algorithms

Most federated learning algorithms can be divided into four components: a *client*, a *server*, an *algorithm*, and a *trainer*.

### Component Overview

Plato still separates workloads into *client*, *server*, *algorithm*, and
*trainer* components, but the client, server, and trainer runtime is now driven
by strategies instead of deep inheritance chains. The defaults reproduce the
legacy behaviour, while custom strategies let you swap individual pieces.

**Client Component:** `plato.clients.base.Client` owns a
`ComposableClient` that orchestrates lifecycle, payload, training, reporting,
and communication strategies. `plato.clients.simple.Client` wires the default
stack; subclass it (or the base client) only to call `_configure_composable(...)`
with your own strategy instances or to add callback plumbing.

**Server Component:** `plato.servers.base.Server` provides lifecycle management
and delegates aggregation plus client selection to strategy objects. Concrete
servers such as `plato.servers.fedavg.Server` accept
`aggregation_strategy=` and `client_selection_strategy=` keyword arguments,
letting you mix built-in logic with custom implementations at runtime.

**Algorithm Component:** Algorithm-specific coordination still lives under
`plato/algorithms/`. Most implementations subclass the relevant algorithm base
class (for example `fedavg.Algorithm`) to integrate with the shared trainer and
payload pipeline.

**Trainer Component:** Trainers mirror the client/server composition model.
`plato.trainers.basic.Trainer` and `plato.trainers.composable.ComposableTrainer`
let you inject loss, optimiser, scheduler, and evaluation behaviour via strategy
objects rather than overriding protected hooks.

??? note "Need more detail?"
    See the **Clients**, **Servers**, and **Trainers** sections in the API
    reference for complete strategy lists and extension hooks. Those pages also
    document when subclassing remains appropriate.

### Implementation Examples

Once the custom *client*, *server*, *algorithm*, and *trainer* pieces have been
implemented, compose them by wiring strategy instances or factories together:

```python
from functools import partial

from plato.clients import simple
from plato.clients.strategies import (
    DefaultCommunicationStrategy,
    DefaultLifecycleStrategy,
    DefaultPayloadStrategy,
    DefaultReportingStrategy,
    DefaultTrainingStrategy,
)
from plato.servers import fedavg
from plato.servers.strategies.aggregation import FedNovaAggregationStrategy
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.training_step import GradientClippingStepStrategy


class AugmentedPayloadStrategy(DefaultPayloadStrategy):
    def outbound_ready(self, context, report, outbound_payload):
        super().outbound_ready(context, report, outbound_payload)
        report.extra_metrics = context.metadata.get("custom_metrics", {})


class CustomClient(simple.Client):
    def __init__(self, *, trainer_factory):
        super().__init__(trainer=trainer_factory)
        self._configure_composable(
            lifecycle_strategy=DefaultLifecycleStrategy(),
            payload_strategy=AugmentedPayloadStrategy(),
            training_strategy=DefaultTrainingStrategy(),
            reporting_strategy=DefaultReportingStrategy(),
            communication_strategy=DefaultCommunicationStrategy(),
        )


trainer_factory = partial(
    ComposableTrainer,
    training_step_strategy=GradientClippingStepStrategy(max_norm=1.0),
)

client = CustomClient(trainer_factory=trainer_factory)
server = fedavg.Server(
    aggregation_strategy=FedNovaAggregationStrategy(),
)

server.run(client)
```

The example above:

- injects a gradient-clipping training step through the trainer factory each
  time a client is instantiated,
- subclasses the simple client to swap only the payload strategy while leaving
  the remaining defaults untouched, and
- swaps the server aggregation logic for FedNova without touching the server
  subclass itself.

Most examples under `examples/` now follow this pattern—copy a nearby strategy,
customise the hook you need, register it in a factory or config, and keep the
rest of the stack unchanged.

---

## Implementing Custom Models and Data Sources

To define a custom model, one does not need to subclass from any base class in Plato. Instead, Plato uses standard model classes in PyTorch.

As shown in `examples/basic/basic.py`, one can define a simple model as follows:

```python
from functools import partial

model = partial(
        nn.Sequential,
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
```

??? note "Partial Functions"
    Since the model will need to be instantiated within Plato itself with `model()`, it should be provided as a partial function using `functools.partial`.

---

If a custom `DataSource` is needed for a custom training session, one can subclass from the `base.DataSource` class.

Example excerpt from `examples/custom_model.py`:

```python
from pathlib import Path
from plato.config import Config

class DataSource(base.DataSource):
    """A custom datasource with custom training and validation
       datasets.
    """
    def __init__(self):
        super().__init__()

        Config()
        base_path = Path(Config.params.get("base_path", "./runtime"))
        data_dir = Path(Config.params.get("data_path", base_path / "data"))
        self.trainset = MNIST(str(data_dir),
                              train=True,
                              download=True,
                              transform=ToTensor())
        self.testset = MNIST(str(data_dir),
                             train=False,
                             download=True,
                             transform=ToTensor())
```

Then, a `DataSource` object can be initialized and passed to the client, along with a custom model and a custom trainer if desired:

```python
model = Model
datasource = DataSource
trainer = Trainer
client = simple.Client(model=model, datasource=datasource, trainer=trainer)
```
