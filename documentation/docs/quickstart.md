# Quick Start

## Running Plato directly using `uv`

To start a federated learning training workload with only a configuration file, run `uv run [Python file] -c [configuration file] ...`. For example:

```bash
uv run plato.py -c configs/MNIST/fedavg_lenet5.yml
```

The following command-line parameters are supported:

- `-c`: the path to the configuration file to be used. The default is `config.yml` in the project's home directory.

- `-b`: the base path, to be used to contain all models, datasets, checkpoints, and results.

- `-r`: resume a previously interrupted training session (only works correctly in synchronous training sessions).

- `-d`: download the dataset to prepare for a training session.

- `--cpu`: use the CPU as the device only.

_Plato_ uses the YAML format for its configuration files to manage runtime configuration parameters. Example configuration files have been provided in the `configs/` directory.

In `examples/`, a number of federated learning algorithms have been included. To run them, just run the main Python program in each of the directories with a suitable configuration file. For example, to run the `basic` example located at `examples/basic/`, run the command:

```bash
uv run examples/basic/basic.py -c configs/MNIST/fedavg_lenet5.yml
```

## Running Plato in a Docker container

To build such a Docker image, use the provided `Dockerfile`:

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

The provided `Dockerfile` helps to build a Docker image running Ubuntu 24.04, with a virtual environment called `plato` pre-configured to run Plato.
