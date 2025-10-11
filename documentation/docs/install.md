# Installation

Plato uses `uv` as its package manager, which is a modern, fast Python package manager that provides significant performance improvements over `conda` environments. To install `uv`, refer to its [official documentation](https://docs.astral.sh/uv/getting-started/installation/), or simply run the following commands:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

To upgrade `uv`, run the command:

```
uv self update
```

To start working with Plato, first clone its git repository:

```bash
git clone git@github.com:TL-System/plato.git
cd plato
```

You can run Plato using `uv run`, using one of its configuration files:

```bash
uv run plato.py -c configs/MNIST/fedavg_lenet5.yml
```

In order to run any of the examples, first run the following command to include all global Python packages in a local Python environment:

```bash
uv sync
```

and then run each example in its own respective directory. For example:

```bash
cd examples/server_aggregation/fedatt
uv run fedatt.py -c fedatt_FashionMNIST_lenet5.yml
```

This will make sure that any additional Python packages, specified in the local `pyproject.yaml` configuration, will be installed first.
