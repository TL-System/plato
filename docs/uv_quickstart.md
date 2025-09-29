# Quickstart

## Activate uv
First of all, make sure the uv environment is active:
```shell
uv sync
source .venv/bin/activate
```

## Running Plato using a configuration file
To start a federated learning training workload, run `uv run [Python file] -c [configuration file] ...`. For example:

```shell
uv run plato.py -c configs
```
- `-c`: the path to the configuration file to be used. The default is `config.yml` in the project's home directory.
- `-b`: the base path, to be used to contain all models, datasets, checkpoints, and results.
- `-r`: resume a previously interrupted training session (only works correctly in synchronous training sessions).
- `-d`: download the dataset to prepare for a training session.
- `--cpu`: use the CPU as the device only.


_Plato_ uses the YAML format for its configuration files to manage the runtime configuration parameters. Example configuration files have been provided in the `configs` directory.

In `examples/`, a number of research projects that were developed using Plato as the federated learning framework have been included. To run them, just run the main Python program in each of the directories with a suitable configuration file. For example, to run the basic project examples/basic/basic.py, run the command:
```shell
uv run examples/basic/basic.py -c examples/outdated/fedrep/fedrep_MNIST_lenet5.yml
```
Here is another example:
```shell
uv run examples/customized_client_training/feddyn/feddyn.py -c examples/customized_client_training/feddyn/feddyn_MNIST_lenet5.yml
```
