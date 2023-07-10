# Quickstart

## Running Plato using a configuration file

To start a federated learning training workload, run `./run` from the repository's root directory. For example:

```shell
./run -c configs/MNIST/fedavg_lenet5.yml
```

- `-c`: the path to the configuration file to be used. The default is `config.yml` in the project's home directory.
- `-b`: the base path, to be used to contain all models, datasets, checkpoints, and results.
- `-r`: resume a previously interrupted training session (only works correctly in synchronous training sessions).
- `-d`: download the dataset to prepare for a training session.
- `--cpu`: use the CPU as the device only.

_Plato_ uses the YAML format for its configuration files to manage the runtime configuration parameters. Example configuration files have been provided in the `configs` directory.

## Running examples built with Plato

In `examples/`, a number of research projects that were developed using Plato as the federated learning framework have been included. To run them, just run the main Python program in each of the directories with a suitable configuration file. For example, to run the FedRep project which implements the FedRep algorithm, run the command:

```shell
python examples/FedRep/fedrep.py -c examples/FedRep/fedrep_MNIST_lenet5.yml
```

## Running Plato with MindSpore or TensorFlow

Plato is designed to support multiple deep learning frameworks, including PyTorch, TensorFlow, and MindSpore.

**TensorFlow.** Install the `tensorflow` and `tensorflow-datasets` pip packages first:

```shell
pip install tensorflow tensorflow-datasets
./run -c configs/MNIST/fedavg_lenet5_tensorflow.yml
```

On macOS with M1 or M2 CPUs, install the `tensorflow-macos` package, rather than `tensorflow`:

```shell
pip install tensorflow-macos tensorflow-datasets
```

**MindSpore.** Plato currently supports the latest MindSpore release, 2.0.0. Follow the installation instructions in the [official MindSpore website](https://mindspore.cn/install/en) to install MindSpore in your conda environment. For example, on an M1 Mac, use the command:

```shell
conda install mindspore=2.0.0 -c mindspore -c conda-forge
```

To use trainers and servers based on MindSpore, assign `true` to `use_mindspore` in the `trainer` section of the configuration file. If GPU is not available when MindSpore is used, assign `true` to `cpuonly` in the `trainer` section as well. These variables are unassigned by default, and _Plato_ would use PyTorch as its default framework. As examples of using MindSpore as its underlying deep learning framework, two configuration files have been provided: `configs/MNIST/fedavg_lenet5_mindspore.yml` and `configs/MNIST/mistnet_lenet5_mindspore.yml`. For example:

```shell
./run -c configs/MNIST/fedavg_lenet5_mindspore.yml
```

## Running Plato in a Docker container

Most of the codebase in _Plato_ is designed to be framework-agnostic, so that it is relatively straightfoward to use _Plato_ with a variety of deep learning frameworks beyond PyTorch, which is the default framwork it is using. One example of such deep learning frameworks that _Plato_ currently supports is [MindSpore 2.0.0](https://www.mindspore.cn/en).

To build such a Docker image, use the provided `Dockerfile` for PyTorch and `Dockerfile_MindSpore` for MindSpore:

```shell
docker build -t plato -f Dockerfile .
```

or:

```shell
docker build -t plato -f Dockerfile_MindSpore .
```

**Note:** Both Dockerfiles assume the use of macOS and Apple M1/M2 CPUs as the host platform. If you use Intel CPUs, replace:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh
```

with

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86.sh -O ~/miniconda3/miniconda.sh
```

To run the docker image that was just built, use the command:

```shell
./dockerrun.sh
```

Or if GPUs are available, use the command:

```shell
./dockerrun_gpu.sh
```

To remove all the containers after they are run, use the command:

```shell
docker rm $(docker ps -a -q)
```

To remove the `plato` Docker image, use the command:

```shell
docker rmi plato
```

On Ubuntu Linux, you may need to add `sudo` before these `docker` commands.

The provided `Dockerfile` helps to build a Docker image running Ubuntu 20.04, with a virtual environment called `plato` pre-configured to support PyTorch 2.0.1 and Python 3.9.17.

If MindSpore support is needed, the provided `Dockerfile_MindSpore` contains two pre-configured environments for CPU and GPU environments, respectively, called `plato_cpu` or `plato_gpu`. They support [MindSpore 2.0.0](https://github.com/mindspore-ai/mindspore) and Python 3.9.17 (which is the Python version that MindSpore 2.0.0 requires). Both Dockerfiles have GPU support enabled. Once an image is built and a Docker container is running, one can use Visual Studio Code to connect to it and start development within the container.
