
## Design principles from a software engineering perspective

This framework is designed to be extensible, making it easy to add new datasets, models, optimizers, hyperparameters, and other customizations. This document discusses the current design of the framework from a software engineering perspective.

This framework makes extensive use of object oriented subclassing with the help of Python 3's [ABC library](https://docs.python.org/3/library/abc.html). It is a good idea to review Python 3's support for base classes with abstract methods before proceeding.

### Hyperparameters

For simplicity, all hyperpameters in the framework are read from a configuration file at the beginning, and stored in a simple `config` object of the `Config` class (found in `config.py`). This includes hyperparameters specific to the dataset, data distribution, and the training process. The command-line arguments are only used to specify the location of the configuration file and the logging level.

### Modules for Datasets, Models, Training

This framework breaks commonly shared components in a federated learning training workload into distinct modules that are as independent as possible.

### The Datasets Module

Each dataset consists of two abstractions:

1. A `Dataset` that stores the dataset, labels, and any data augmentation.

2. A `Divider` that partitions the dataset for local training or testing at each client in the federated learning workload.

For now, we use the standard PyTorch `DataLoader` class in `torch.utils.data` for loading data. We may need to write our own custom data loaders in the future (*to be completed*). This may be useful for ImageNet (`datasets/imagenet.py`), which replaces all functionality due to the specialized needs of loading such a large dataset efficiently (*to be completed*).

A dataset must subclass the `Dataset` abstract base classes in `datasets/base.py`. This class subclasses the corresponding PyTorch `Dataset` class, and adds additional functionality to support build-in transformations.

The external interface of this module is contained in `datasets/registry.py`. The registry contains a list of all existing datasets in the framework (so that they can be discovered and loaded). Its most important function is `get()`, which returns a `DataSet` object.

### The Models Module

Each model is created by subclassing the `Model` abstract base class in `models/base.py`. This base class is a valid PyTorch `nn.Module` with several additional abstract methods that support other functionality throughout the framework. In particular, any subclass must have static methods to determine whether a string model name (e.g., `cifar_resnet_18`) is valid and to create a model object from a string name, a number of outputs, and an initializer.

The external interface of this module is contained in `models/registry.py`. Just like `datasets/registry.py`, there is a list of all existing models in the framework so that they can be discovered and loaded. The registry similarly contains a `get()` function that returns the corresponding `Model` as specified. 

Alternatively, rather than writing our own custom registry, it is conceivable to use a Python package called `ClassRegistry`, as it supports both the *registry* and the *factory* design pattern. However, `ClassRegistry` only supports the use of one name for each class, while in our case we may need to have multiple names (representing corresponding variants) for each class. An example of this can be found in `models/cifar_resnet.py`, which supports four different variants of `ResNet`.

### The Training Module

The training module centers on a single function: the `train()` function in `training/train.py`. This function takes a `Model`, a `DataSet`, and a `config` as its arguments (*to be completed*). It then trains the `Model` on the dataset, as specified in the training hyperparameters in `config`.

The training module would use callbacks for customization (*to be completed*).

To create optimizers and learning rate scheduler objects, `train()` calls the `get_optimizer()` and `get_lr_schedule()` functions (*to be completed*) in `training/optimizers.py`, which serve as small-scale registries for these objects.
