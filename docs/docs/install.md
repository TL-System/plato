# Installation

Use Python 3.13 and [uv](https://docs.astral.sh/uv/getting-started/installation/)
to install Plato from its checked-in manifest and lockfile. The repository's
build and CI tools use uv 0.12.22. To install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Clone the repository and provision its environment:

```bash
git clone https://github.com/TL-System/plato.git
cd plato
uv sync --locked --python 3.13
```

Run a reference workload from the repository root:

```bash
uv run --locked python plato.py --config configs/MNIST/fedavg_lenet5.toml --cpu
```

See [Quick Start](quickstart.md) for device selection, output paths and Docker
launch commands.

`uv sync` installs the selected project dependencies into `.venv`; it does not
copy all globally installed Python packages. Optional **extras** select runtime
features with `--extra`, while **dependency groups** select tooling or test
requirements with `--group` (for example, `--group docs`). Select only the extras
needed by a workload:

```bash
uv sync --locked --python 3.13 --extra mlx
```

The `ssl` and `llm_eval` extras are declared incompatible. `uv sync --all-extras`
fails for this project; use separate environments for self-supervised workloads
and Lighteval. For example, provision SSL separately with:

```bash
UV_PROJECT_ENVIRONMENT=.venv-ssl uv sync --locked --python 3.13 --extra ssl
```

Use the same environment selection and extra when running that workload.

Useful extras in the current root package include:

- `llm_eval` for server-side Lighteval evaluation
- `mlx` for Apple Silicon MLX workloads
- `dp`, `rl`, and `mpc` for specialized research workloads

Most example directories use the root project dependencies. Follow each
example's instructions for its entrypoint and config; for example:

```bash
cd examples/server_aggregation/fedatt
uv run --locked python fedatt.py -c fedatt_FashionMNIST_lenet5.toml
```

Some examples have their own `pyproject.toml` and are registered as workspace
members in the root manifest. Running uv from one of those directories selects
that member and its additional dependencies. A directory without its own
manifest does not gain extra packages just by changing into it.

### Optional: MLX Backend for Apple Silicon

The native MLX reference is LeNet-5/MNIST with FedAvg on Apple Silicon. Install
its optional extra with the default Python 3.13 interpreter:

```bash
uv sync --python 3.13 --extra mlx
```

Use `uv run --extra mlx` when launching the workload. See [Native MLX](mlx.md)
for runnable commands, supported controls, and native CPU/Metal qualification.
The normal Linux core CI suite does not qualify this backend.

### Optional: Server-side LLM Evaluation with Lighteval

To enable `evaluation.type = "lighteval"`, install the locked evaluator stack
and provision the two NLTK tokenizer resources used by its task registry:

```bash
uv sync --locked --python 3.13 --extra llm_eval
uv run --locked --extra llm_eval python -m nltk.downloader punkt punkt_tab
```

Provision `punkt` and `punkt_tab` before offline use; they are data resources,
not Python packages. The evaluator extra includes `langdetect` and Lighteval.
Keep `--extra llm_eval` on **every syncing `uv run` command** for evaluation.
A later plain `uv run` can select the default shared dependencies, including
an incompatible xxhash major version, even when Lighteval remains installed.
The extra constrains xxhash to the compatible 3.x line. Use a direct environment
interpreter or `uv run --no-sync` only after that environment has been provisioned
with the compatible locked extra; neither command repairs a changed environment.

See:

- [Evaluation](configurations/evaluation.md) for the configuration contract
- [Server-side Lighteval for SmolLM2](examples/case-studies/4. Server-side Lighteval for SmolLM2.md) for an end-to-end example

### Qwen3 Federated LoRA

The Qwen3 reference uses the standard Hugging Face and PEFT dependencies from
`uv sync`. Use Python 3.13, the default qualification and CI target. See
[Qwen3 Federated LoRA](examples/case-studies/6. Qwen3 Federated LoRA.md)
for the pinned model, local data, CPU command, and validation scope.

### Building the Documentation

From the repository root with Python 3.13 available, run:

```bash
PLATO_DOCS_PYTHON="$(uv python find 3.13)" ./docs/build.sh
```

The script uses uv 0.12.22, bootstrapping it in `.venv-docs-bootstrap` if needed.
It provisions the locked docs-only group in `.venv-docs`, checks the generated
`docs/requirements.txt` against the lockfile, and runs a strict MkDocs build.
The HTML output is in `docs/site`. This environment does not install Plato,
PyTorch or Lighteval. Set `PLATO_DOCS_ENVIRONMENT` to use another dedicated docs
environment; the application `.venv` is not a valid destination.

### Building the `plato-learn` PyPI Package

With uv 0.12.22 installed, run the same package check used by the release
workflow from a committed checkout:

```bash
uv run --no-project --python 3.13 python .github/scripts/check_distribution.py
```

The check compares archive source bytes with Git HEAD, so commit tracked source
changes before running it. Generated build outputs are excluded.

This builds the wheel and source distribution with build dependencies constrained
from `uv.lock`, checks their contents, rebuilds a wheel from the source archive,
and installs the wheel in an isolated Python 3.13 environment for import and CPU
operation checks. It downloads the required build and runtime dependencies.
Distributions, logs and validation receipts go to `ci-artifacts/docs-package`.
That output directory must not already exist; use `--output-dir` to select a new
path outside the repository or beneath `ci-artifacts` for another run.

The check does not publish a package. The release-created GitHub workflow is
configured to publish its validated distributions using the repository's PyPI
token.

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
