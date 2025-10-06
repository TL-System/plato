# Quick Start

To start a federated learning training workload, run `uv run [Python file] -c [configuration file] ...`. For example:

```shell
uv run plato.py -c configs/MNIST/fedavg_lenet5.yml
```

The following command-line parameters are supported:

- `-c`: the path to the configuration file to be used. The default is `config.yml` in the project's home directory.

- `-b`: the base path, to be used to contain all models, datasets, checkpoints, and results.

- `-r`: resume a previously interrupted training session (only works correctly in synchronous training sessions).

- `-d`: download the dataset to prepare for a training session.

- `--cpu`: use the CPU as the device only.

_Plato_ uses the YAML format for its configuration files to manage runtime configuration parameters. Example configuration files have been provided in the `configs/` directory.

In `examples/`, a number of research projects that were developed using Plato as the federated learning framework have been included. To run them, just run the main Python program in each of the directories with a suitable configuration file. For example, to run the basic project examples/basic/basic.py, run the command:

```shell
uv run examples/basic/basic.py -c configs/MNIST/fedavg_lenet5.yml
```

Here is another example:

```shell
uv run examples/customized_client_training/feddyn/feddyn.py -c examples/customized_client_training/feddyn/feddyn_MNIST_lenet5.yml
```

# Running Plato in a Docker container
Most of the codebase in Plato is designed to be framework-agnostic, so that it is relatively straightfoward to use Plato with a variety of deep learning frameworks beyond PyTorch, which is the default framwork it is using.

To build such a Docker image, use the provided Dockerfile for PyTorch:

```shell
docker build -t plato -f Dockerfile .
```

To run the docker image that was just built, use the command:
```shell
# Make sure it's executable
chmod +x dockerrun.sh

# Run the container
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

To remove the plato Docker image, use the command:
```shell
docker rmi plato
```

The provided `Dockerfile` helps to build a Docker image running Ubuntu 24.04, with a virtual environment called `plato` pre-configured to support PyTorch 2.8.0 and Python 3.12.6.
