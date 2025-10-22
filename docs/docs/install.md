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
uv run plato.py -c configs/MNIST/fedavg_lenet5.toml
```

In order to run any of the examples, first run the following command to include all global Python packages in a local Python environment:

```bash
uv sync
```

In case you need extra dependencies (including homormorphic encryption, differential privacy, reinforcement learning, secure multi-party computation, self-supervised learning, HuggingFace transformers and PEFT, or the MLX framework), you should run:

```bash
uv sync --all-extras
```

or:

```bash
uv sync --extra mlx
```

where `mlx` is the name of the dependency group.

Each example should be run in its own directory:

```bash
cd examples/server_aggregation/fedatt
uv run fedatt.py -c fedatt_FashionMNIST_lenet5.toml
```

This will make sure that any additional Python packages, specified in the local `pyproject.toml` configuration, will be installed first.

### Optional: MLX Backend for Apple Silicon

To use MLX as a backend alternative to PyTorch on Apple Silicon devices, install the MLX dependencies:

```bash
uv sync --extra mlx
```

See the [Quick Start guide](quickstart.md#using-mlx-as-a-backend) for configuration details.

### Building the `plato-learn` PyPi Package

The `plato-learn` PyPi package will be automatically built and published by a GitHub action workflow every time a release is created on GitHub. To build the package manually, follow these steps:

1. Clean previous builds (optional):
```bash
rm -rf dist/ build/ *.egg-info
```

2. Build the package:
```bash
uv build
```

3. Publish to PyPI:
    ```bash
    uv publish
    ```

    Or if you need to specify the PyPi token explicitly:
    ```bash
    uv publish --token <your-pypi-token>
    ```

The `uv` tool will handle all the build process using the modern, PEP 517-compliant `hatchling` backend specified in `pyproject.toml`, making it much simpler than the old `python setup.py sdist bdist_wheel` approach.

### Uninstalling Plato

Plato can be uninstalled by simply removing the local environment, residing within the top-level directory:

```bash
rm -rf .venv
```

Optionally, you may also clean `uv`’s cache:

```bash
uv cache clean
```

Optionally, you can also uninstall `uv` itself by following the [official uv documentation](https://docs.astral.sh/uv/getting-started/installation/#uninstallation).
