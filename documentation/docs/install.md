# Installation

Plato uses `uv` as its package manager, which is a modern, fast Python package manager that provides significant performance improvements over `conda` environments. To install `uv`, refer to its [official documentation](https://docs.astral.sh/uv/getting-started/installation/), or simply run the following commands:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

To upgrade `uv`, run the command:

```
uv self update
```

To start working with Plato, first clone its git repository:

```shell
git clone git@github.com:TL-System/plato.git
cd plato
```

You can then run Plato, including its examples, using `uv run` directly. For example:

```shell
uv run plato.py -c configs/MNIST/fedavg_lenet5.yml
```
