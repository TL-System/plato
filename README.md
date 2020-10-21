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

### Simulation

To start a simulation, run [`run.py`](run.py) from the repository's root directory:

```shell
cp configs/MNIST/mnist.conf config.conf
python run.py
  --config=config.conf
  --log=INFO
```

* `--config` (`-c`): the path to the configuration file to be used. The default is `config.conf` in the project's home directory.
* `--log` (`-l`): the level of logging information to be written to the console, defaults to `INFO`.

*Plato* uses a standard configuration file, parsed by Python's standard configuration parser, to manage the runtime configuration parameters. Example configuration files have been provided in the `configs` directory.

### Uninstalling Plato

Remove the `conda` environment used to run *Plato* first, and then remove the directory containing *Plato*'s git repository.

```shell
conda-env remove -n federated
rm -rf plato/
```

where `federated` is the name of the `conda` environment that *Plato* runs in.

### Technical support

Technical support questions should be directed to the maintainer of this software framework: Baochun Li (bli@ece.toronto.edu).
