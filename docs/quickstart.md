# Quickstart

## Running Plato using a configuration file

To start a federated learning training workload, run `./run` from the repository's root directory. For example:

```bash
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

```bash
python examples/FedRep/fedrep.py -c examples/FedRep/fedrep_MNIST_lenet5.yml
```

## Running Plato in a Docker container

To build a new Docker image, use the provided `Dockerfile` for PyTorch:

```bash
docker build -t plato -f Dockerfile .
```

To run the docker image that was just built, use the command:

```bash
./dockerrun.sh
```

Or if GPUs are available, use the command:

```bash
./dockerrun_gpu.sh
```

To remove all the containers after they are run, use the command:

```bash
docker rm $(docker ps -a -q)
```

To remove the `plato` Docker image, use the command:

```bash
docker rmi plato
```

On Ubuntu Linux, you may need to add `sudo` before these `docker` commands.

The provided `Dockerfile` helps to build a Docker image running Ubuntu 20.04, with a virtual environment called `plato` pre-configured to support PyTorch 2.0.1 and Python 3.9.17.
