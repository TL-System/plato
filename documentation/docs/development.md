
# Developer's Guide

The Plato framework is designed to be extensible, hopefully making it easy to add new data sources for datasets, models, and custom trainers for models. This document discusses the current design of the framework from a software engineering perspective.

This framework makes extensive use of object oriented subclassing with the help of Python 3's [ABC library](https://docs.python.org/3/library/abc.html). It is a good idea to review Python 3's support for base classes with abstract methods before proceeding. It also makes sporadic use of Python 3's [Data Classes](https://docs.python.org/3/library/dataclasses.html). It also supports defining callback classes, and customizing a trainer by providing it with a list of custom callback classes.

---

## Configuration Parameters

All configuration parameters are globally accessed using the Singleton `Config` class (found in `config.py`). They are read from a configuration file when the clients and the servers launch, and the configuration file follows the YAML format for the sake of simplicity and readability.

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

**Client Component:** The *client* implements all algorithm logic on the client side. Typically, one would subclass from the `simple.Client` class to reuse some of the useful methods there, but it is also possible to subclass from the `base.Client` class.

**Server Component:** The *server* implements all algorithm logic on the server side. Typically, one would subclass from the `fedavg.Server` class to reuse some of the useful methods there, but it is also possible to subclass from the `base.Server` class.


**Algorithm Component:** Framework-specific algorithm logic should be implemented in an *algorithm* module. Typically, one would subclass from the `fedavg.Algorithm` class. Several frequently-used algorithms are provided in `algorithms/`, while more examples are provided outside the framework in `examples/`.

**Trainer Component:** Custom training loops should be implemented as a *trainer* class. If a trainer is to be implemented, one may subclass from the `basic.Trainer` class.

??? note "Note"
    See the **Trainers** section in API reference documentation for customizing the training loops using inheritance or callbacks.

### Implementation Examples

Once the custom *client*, *server*, *algorithm*, *trainer* classes have been implemented, they can be initialized just like the following examples:

From `examples/basic/basic.py`:

```python
model = Model
datasource = DataSource
trainer = Trainer
client = simple.Client(model=model, datasource=datasource, trainer=trainer)
server = fedavg.Server(model=model, datasource=datasource, trainer=trainer)
server.run(client)
```

Example from `examples/FedRep/fedrep.py`:

```python
trainer = fedrep_trainer.Trainer
algorithm = fedrep_algorithm.Algorithm
client = fedrep_client.Client(algorithm=algorithm, trainer=trainer)
server = fedrep_server.Server(algorithm=algorithm, trainer=trainer)

server.run(client)
```

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
class DataSource(base.DataSource):
    """A custom datasource with custom training and validation
       datasets.
    """
    def __init__(self):
        super().__init__()

        self.trainset = MNIST("./data",
                              train=True,
                              download=True,
                              transform=ToTensor())
        self.testset = MNIST("./data",
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
