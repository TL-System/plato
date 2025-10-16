# Repository Guidelines

## Project Structure & Module Organization
Plato's core runtime lives in `plato/`. Key submodules include `algorithms/` (federated strategies), `clients/` plus `client.py` (orchestration logic), `servers/` (coordination), `trainers/` (model/optimizer loops), and supporting layers in `datasources/`, `samplers/`, `processors/`, and `utils/`. Scenario definitions sit under `configs/<dataset>/` as YAML, while reproducible research case studies live in `examples/` with their own workspace entries described in `pyproject.toml`. Shared documentation is under `docs/`, and automated checks are centralized in `tests/`.

## Build, Test, and Development Commands
- `uv sync` provisions dependencies declared in `pyproject.toml` and `uv.lock`.
- `source .venv/bin/activate` to activate the Python virtual environment after `uv sync`.
- `uv run python plato.py --config configs/MNIST/fedavg_lenet5.yml` launches a reference experiment; swap in different config paths as needed.
- `uv run pytest` runs the complete suite; scope it with a target like `tests/test_strategies_simple.py` for faster feedback.
- `uv run ruff check . --select I --fix` enforces the repo's import-order policy before creating a pull request.
- When running any shell commands, use `zsh -lc` to load the current `.zshrc` environment.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and keep lines ≤88 characters (matching the Ruff configuration). Package and module names stay lowercase with underscores; classes use PascalCase; functions, methods, and variables use snake_case. Prefer explicit type hints on new public APIs, and keep docstrings concise but descriptive. YAML configuration keys should remain lowercase_with_underscores and align to the indentation already present in `configs/`.

## Testing Guidelines
Pytest is the standard harness. Add focused unit tests alongside the closest existing coverage (e.g., new server logic belongs near `tests/test_strategies_integration.py`). Integration scenarios should exercise their YAML configs via lightweight smoke tests when practical. Use deterministic seeds (see `server.random_seed` in configs) to make failures reproducible, and include negative-path or degradation checks whenever behaviour may regress existing algorithms.

## Commit & Pull Request Guidelines
Match the Git history by writing sentence-style commit subjects with initial capitals and closing periods (e.g., “Refactored FedBuff for the new strategy API.”). Group related changes per commit, and include a brief body if context is non-obvious. Pull requests should outline the motivation, enumerate config or interface changes, link tracking issues, and attach experiment metrics or logs when altering training behaviour. Mention required follow-up work explicitly to aid other contributors.

## Configuration Tips
Prefer copying an existing YAML from `configs/` when proposing a new scenario, updating only the deltas. Keep shared processors reusable by placing them in `plato/processors/`, and register new workspace modules inside the `tool.uv.workspace.members` list so dependency resolution stays deterministic.
