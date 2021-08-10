
## Design

The Plato framework is designed to be extensible, hopefully making it easy to add new data sources for datasets, models, and custom trainers for models. This document discusses the current design of the framework from a software engineering perspective.

This framework makes extensive use of object oriented subclassing with the help of Python 3's [ABC library](https://docs.python.org/3/library/abc.html). It is a good idea to review Python 3's support for base classes with abstract methods before proceeding. It also makes sporadic use of Python 3's [Data Classes](https://docs.python.org/3/library/dataclasses.html).

### Configuration parameters

All configuration parameters are globally accessed using the Singleton `Config` class globally (found in `config.py`). They are read from a configuration file when the clients and the servers launch, and the configuration file follows the YAML format for the sake of simplicity and readability. These parameters include parameters specific to the dataset, data distribution, trainer, the federated learning algorithm, server configuration, and cross-silo training.

Either a command-line argument (`-c` or `--config`) or an environment variable `config_file` can be used to specify the location of the configuration file. Use `Config()` anywhere in the framework to access these configuration parameters.

### Extensible modules

This framework breaks commonly shared components in a federated learning training workload into extensible modules that are as independent as possible.

#### Data sources

A `Datasource` instance is used to obtain the dataset, labels, and any data augmentation. For example, the PyTorch `DataLoader` class in `torch.utils.data` can be used to load the MNIST dataset; `Datasets` classes in the `HuggingFace` framework can also be used as a data source to load datasets.

A data source must subclass the `Datasource` abstract base classes in `datasources/base.py`. This class may use third-party frameworks to load datasets, and may add additional functionality to support build-in transformations.

The external interface of this module is contained in `datasources/registry.py`. The registry contains a list of all existing datasources in the framework, so that they can be discovered and loaded. Its most important function is `get()`, which returns a `DataSource` instance.

#### Samplers 

A `Sampler` is responsible for sampling a dataset for local training or testing at each client in the federated learning workload. This is used to *simulate* a local dataset that is available locally at the client, using either an i.i.d. or non-i.i.d. distribution. For non-i.i.d. distributions, an example sampler that is based on the Dirichlet distribution (with a configurable concentration bias) is provided. Samplers are passed as one of the parameters to a PyTorch `Dataloader` or MindSpore `Dataset` instance.

#### Models

Each model is created by subclassing the `Model` abstract base class in `models/base.py`. This base class is a valid PyTorch `nn.Module` with several additional abstract methods that support other functionality throughout the framework. In particular, any subclass must have static methods to determine whether a string model name (e.g., `cifar_resnet_18`) is valid and to create a model object from a string name, a number of outputs, and an initializer.

The external interface of this module is contained in `models/registry.py`. Just like `datasets/registry.py`, there is a list of all existing models in the framework so that they can be discovered and loaded. The registry similarly contains a `get()` function that returns the corresponding `Model` as specified. 

### Extending Plato with new federated learning algorithms

Most federated learning algorithms can be divided into four components: a *client*, a *server*, an *algorithm*, and a *trainer*.

- The *client* implements all algorithm logic on the client side. Typically, one would inherit from the `simple.Client` class to reuse some of the useful methods there, but it is also possible to inherit from the `base.Client` class.

- The *server* implements all algorithm logic on the server side. Typically, one would inherit from the `fedavg.Server` class to reuse some of the useful methods there, but it is also possible to inherit from the `base.Server` class.

    *Note:* Implementations for both the client and the server should be neutral across various deep learning frameworks, such as PyTorch, TensorFlow, and MindSpore.

- Framework-specific algorithm logic should be implemented in an *algorithm* module. Typically, one would inherit from the PyTorch-based `fedavg.Algorithm` class if PyTorch is to be used. If other frameworks, for example TensorFlow, is to be used, one can inherit from the `tensorflow.fedavg.Algorithm` class. Several frequently-used algorithms are provided in `algorithms/`, while more examples are provided outside the framework in `examples/`.

- Custom training loops should be implemented as a *trainer* class. If a PyTorch-based trainer is to be implemented, one may inherit from the `basic.Trainer` class. Typically, the `train_model` method should be overridden with a custom implementation.

Once the custom *client*, *server*, *algorithm*, *trainer* classes have been implemented, they can be initialized using the following example code (from `examples/split_learning`):

```python
trainer = split_learning_trainer.Trainer()
algorithm = split_learning_algorithm.Algorithm(trainer=trainer)
client = split_learning_client.Client(algorithm=algorithm, trainer=trainer)
server = split_learning_server.Server(algorithm=algorithm, trainer=trainer)

server.run(client)
```

### Implementing custom models and data sources

To define a custom model, one does not need to inherit from any base class in Plato, as Plato uses standard model classes in each machine learning framework. For example, Plato uses `nn.Module` as the base class in PyTorch, `nn.Cell` as the base class in MindSpore, and `keras.Model` as the base class in TensorFlow.

For example (excerpt from `examples/custom_model.py`), one can define a simple model in PyTorch as follows:

```python
model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
```

If a custom `DataSource` is also needed for a custom training session, one can inherit from the `base.DataSource` class (assuming PyTorch is used as the framework), as in the following example (excerpt from `examples/custom_model.py`):

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

Then, a `DataSource` object can be initialized and passed to the client, along with a custom model if desired:

```python
datasource = DataSource()
trainer = Trainer(model=model)
client = simple.Client(model=model, datasource=datasource)
```

