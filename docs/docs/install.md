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

In case you need optional dependency groups, you can install them with:

```bash
uv sync --all-extras
```

or:

```bash
uv sync --extra mlx
```

where `mlx` is the name of the dependency group.

Useful extras in the current root package include:

- `llm_eval` for server-side Lighteval evaluation
- `nanochat` for Nanochat training and CORE evaluation
- `mlx` for Apple Silicon MLX workloads
- `dp`, `rl`, and `mpc` for specialized research workloads

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

### Optional: Server-side LLM Evaluation with Lighteval

To enable `evaluation.type = "lighteval"`, install the evaluator stack:

```bash
uv sync --extra llm_eval
```

This installs `lighteval` together with the runtime dependencies used by Plato's built-in Lighteval adapter.

See:

- [Evaluation](configurations/evaluation.md) for the configuration contract
- [Server-side Lighteval for SmolLM2](examples/case-studies/4. Server-side Lighteval for SmolLM2.md) for an end-to-end example

### Optional: Nanochat Training and CORE Evaluation

To use Nanochat workloads or the `nanochat_core` evaluator, install:

```bash
uv sync --extra nanochat
```

Nanochat also requires the `external/nanochat` git submodule, and the Rust tokenizer extension must be built before running the Nanochat configs successfully.

See [Nanochat in Plato](examples/case-studies/5. Nanochat in Plato.md) for the full step-by-step setup, including:

- `git submodule update --init --recursive`
- installing `maturin` and building `rustbpe`
- preparing the tokenizer required by CORE evaluation
- running `configs/Nanochat/synthetic_micro.toml` and `configs/Nanochat/parquet_micro.toml`

### Optional: SmolVLA + LeRobot Robotics Stack

The LeRobot / SmolVLA path is intentionally kept separate from the default Plato install so the root environment stays lean.

!!! warning "Migration note"
    Older runbooks may still reference `uv sync --extra robotics`.
    That root-package extra no longer exists. Use a dedicated environment that already has the LeRobot / SmolVLA stack installed, then verify it with:

    ```bash
    uv run python -c "import lerobot; print(lerobot.__version__)"
    ```

See [SmolVLA Trainer with LeRobot](examples/case-studies/3. SmolVLA Trainer with LeRobot.md) for the current setup guidance, configuration contract, and troubleshooting notes.

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
