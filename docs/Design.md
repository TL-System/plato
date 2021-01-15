
## Design

The Plato framework is designed to be extensible, hopefully making it easy to add new datasets, models, optimizers, hyperparameters, and other customizations. This document discusses the current design of the framework from a software engineering perspective.

This framework makes extensive use of object oriented subclassing with the help of Python 3's [ABC library](https://docs.python.org/3/library/abc.html). It is a good idea to review Python 3's support for base classes with abstract methods before proceeding. It also makes sporadic use of Python 3's [Data Classes](https://docs.python.org/3/library/dataclasses.html).

### Hyperparameters

For simplicity, all hyperpameters in the framework are read from a configuration file at the beginning, and stored in a global Singleton `Config` class (found in `config.py`). This includes hyperparameters specific to the dataset, data distribution, the training process, server configuration, and cross-silo training. The command-line arguments are only used to specify the location of the configuration file, the logging level, the client ID (on the client side), and the port number (for edge servers in cross-silo training). Use `Config()` anywhere in the framework to access these hyperparameters.

In the future, hyperparameters belonging to the framework in general will be separated from hyperparameters belonging to a specific mechanism. Both will continue to use global Singleton classes.

### Modules for Datasets, Models, and Training

This framework breaks commonly shared components in a federated learning training workload into distinct modules that are as independent as possible.

### The Datasets Module

Each dataset consists of two abstractions:

1. A `Dataset` that stores the dataset, labels, and any data augmentation.

2. A `Divider` that partitions the dataset for local training or testing at each client in the federated learning workload.

For now, we use the standard PyTorch `DataLoader` class in `torch.utils.data` for loading data. Custom data loaders are used to support custom datasets that are not supported by the PyTorch `DataLoader` class, such as the CINIC-10 and ImageNet datasets. This is especially the case for ImageNet, due to the specialized needs of loading such a large dataset efficiently (*to be completed*).

A dataset must subclass the `Dataset` abstract base classes in `datasets/base.py`. This class subclasses the corresponding PyTorch `Dataset` class, and adds additional functionality to support build-in transformations.

The external interface of this module is contained in `datasets/registry.py`. The registry contains a list of all existing datasets in the framework (so that they can be discovered and loaded). Its most important function is `get()`, which returns a `DataSet` object.

### The Models Module

Each model is created by subclassing the `Model` abstract base class in `models/base.py`. This base class is a valid PyTorch `nn.Module` with several additional abstract methods that support other functionality throughout the framework. In particular, any subclass must have static methods to determine whether a string model name (e.g., `cifar_resnet_18`) is valid and to create a model object from a string name, a number of outputs, and an initializer.

The external interface of this module is contained in `models/registry.py`. Just like `datasets/registry.py`, there is a list of all existing models in the framework so that they can be discovered and loaded. The registry similarly contains a `get()` function that returns the corresponding `Model` as specified. 

Alternatively, rather than writing our own custom registry, it is conceivable to use a Python package called `ClassRegistry`, as it supports both the *registry* and the *factory* design pattern. However, `ClassRegistry` only supports the use of one name for each class, while in our case we may need to have multiple names (representing corresponding variants) for each class. An example of this can be found in `models/cifar_resnet.py`, which supports four different variants of `ResNet`.

### Implementing federated learning algorithms

Most federated learning algorithms can be divided into three components: a *client*, a *server*, and a *trainer*. The *client* implements all algorithm-specific logic on the client side, but it should remain neutral to deep learning frameworks such as PyTorch, TensorFlow, or MindSpore. The *server* implements all algorithm-specific logic on the server side, but it should also be neutral across various deep learning frameworks. All the algorithm-specific logic that is framework-specific should be included in a *trainer* module, found in `trainers/`.

One of the important functions in the *trainer* module is the `train()` function, which trains the trainer's model on the provided dataset. This function would use callbacks for customization (*to be completed*). To create optimizers and learning rate scheduler objects, `train()` calls the `get_optimizer()` and `get_lr_schedule()` functions in `trainers/optimizers.py`, which serve as small-scale registries for these objects.
