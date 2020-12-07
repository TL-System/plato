## Plato: A New Framework for Federated Learning Research

Welcome to *Plato*, a new software framework to facilitate scalable federated learning research.

### Installation

To install *Plato*, first clone this repository to the desired directory.

*Plato* uses [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage its Python packages. Before using *Plato*, first install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), update your `conda` environment, and then create a new `conda` environment with Python 3.8 using the command:

```shell
$ conda update conda
$ conda create -n federated python=3.8
$ conda activate federated
```

where `federated` is the the preferred name of your new environment.

Update any packages, if necessary by typing `y` to proceed.

The next step is to install the required Python packages. PyTorch should be installed following the advice of its [getting started website](https://pytorch.org/get-started/locally/). The typical command in Linux with CUDA GPU support, for example, would be:

```shell
$ conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

In macOS (without GPU support), the typical command would be:

```shell
$ conda install pytorch torchvision -c pytorch
```

We will need to install the `websockets` package for client-server communication:

```shell
$ pip install websockets
```

Install `matplotlib` package for plotting figures of results:

```shell
$ conda install matplotlib
```

In case unit tests in the `tests` directory need to be run, `scipy` should also be installed:

```shell
$ conda install scipy
```

In case it is needed to format Python code during development, install `yapf`:

```shell
$ pip install yapf
```

If you use Visual Studio Code, to use `yapf`:

```shell
$ vi ~/Library/Application\ Support/Code/user/settings.json
```

And add the following:

```shell
    “python.linting.enabled": true,
    "python.linting.pylintPath": "pylint",
    "editor.formatOnSave": true,
    "python.formatting.provider": "yapf", 
    "python.linting.pylintEnabled": true
```

### Running Plato

To start a federated learning training workload, run [`run.py`](run.py) from the repository's root directory. For example:

```shell
cp configs/MNIST/mnist.conf config.conf
python server.py
  --config=config.conf
  --log=info
```

* `--config` (`-c`): the path to the configuration file to be used. The default is `config.conf` in the project's home directory.
* `--log` (`-l`): the level of logging information to be written to the console. Possible values are `critical`, `error`, `warn`, `info`, and `debug`, and the default is `info`.

*Plato* uses a standard configuration file, parsed by Python's standard configuration parser, to manage the runtime configuration parameters. Example configuration files have been provided in the `configs` directory.

### Running Unit Tests

All unit tests are in the `tests/` directory. These tests are designed to be standalone and executed separately. For example, the command `python lr_schedule_tests.py` runs the unit tests for learning rate schedules.

### Uninstalling Plato

Remove the `conda` environment used to run *Plato* first, and then remove the directory containing *Plato*'s git repository.

```shell
conda-env remove -n federated
rm -rf plato/
```

where `federated` is the name of the `conda` environment that *Plato* runs in.

For more specific documentation on how Plato can be run on GPU cluster environments such as Lambda Labs' GPU cloud or Compute Canada, refer to `docs/Running.md`.

### Technical support

Technical support questions should be directed to the maintainer of this software framework: Baochun Li (bli@ece.toronto.edu).
