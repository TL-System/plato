# Repository Guidelines

## Project Structure & Module Organization
Plato's core runtime lives in `plato/`. Key submodules include `algorithms/` (federated strategies), `clients/` plus `client.py` (orchestration logic), `servers/` (coordination), `trainers/` (model/optimizer loops), and supporting layers in `datasources/`, `samplers/`, `processors/`, and `utils/`. Scenario definitions sit under `configs/<dataset>/` as TOML, while reproducible research case studies live in `examples/` with their own workspace entries described in `pyproject.toml`. Shared documentation is under `docs/`, and automated checks are centralized in `tests/`.

## Build, Test, and Development Commands
- `uv sync --locked --python 3.13` provisions dependencies declared in `pyproject.toml` and `uv.lock`.
- `source .venv/bin/activate` to activate the Python virtual environment after `uv sync`.
- `uv run --locked python plato.py --config configs/MNIST/fedavg_lenet5.toml` launches a reference experiment; swap in different config paths as needed.
- `uv run --locked --group test-model-search python -m pytest tests --test-profile=mandatory -ra` runs the mandatory core suite, including retained model-search tests.
- `uv run --locked --group test-model-search python -m pytest tests/trainers/test_lr_scheduler_registry.py -ra` runs focused scheduler coverage from the repository root.
- `uv run ruff check . --select I --fix` enforces the repo's import-order policy before creating a pull request.
- When running any shell commands, use `zsh -lc` to load the current `.zshrc` environment.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and keep lines ≤88 characters (matching the Ruff configuration). Package and module names stay lowercase with underscores; classes use PascalCase; functions, methods, and variables use snake_case. Prefer explicit type hints on new public APIs, and keep docstrings concise but descriptive. TOML configuration keys should remain lowercase_with_underscores.

## Testing Guidelines
Pytest is the standard harness. Add focused unit tests alongside the closest existing coverage (e.g., new server logic belongs near `tests/servers/test_fedavg_strategy.py`). Integration scenarios should exercise their TOML configs via lightweight smoke tests when practical. Use deterministic seeds (see `server.random_seed` in configs) to make failures reproducible, and include negative-path or degradation checks whenever behaviour may regress existing algorithms.

Ordinary `pytest tests` and the mandatory profile exclude native MLX, Lighteval
runtime, and Phase 4 example qualification directories. Follow the separate
[native MLX checks](docs/docs/mlx.md#run-native-checks),
[Lighteval runtime qualification](docs/docs/references/evaluators.md#optional-runtime-qualification),
and [Phase 4 inventory](tests/examples_phase4/cases.json) with its collection
policy in `tests/conftest.py`. A core suite pass does not qualify those workloads.

## Commit & Pull Request Guidelines
Match the Git history by writing sentence-style commit subjects with initial capitals and closing periods (e.g., “Refactored FedBuff for the new strategy API.”). Group related changes per commit, and include a brief body if context is non-obvious. Pull requests should outline the motivation, enumerate config or interface changes, link tracking issues, and attach experiment metrics or logs when altering training behaviour. Mention required follow-up work explicitly to aid other contributors.

## Configuration Tips
Prefer copying an existing TOML from `configs/` when proposing a new scenario, updating only the deltas. Keep shared processors reusable by placing them in `plato/processors/`, and register new workspace modules inside the `tool.uv.workspace.members` list so dependency resolution stays deterministic.
