# SmolVLA + LeRobot Validation Report (T12)

Date: 2026-02-19  
Validation window: 2026-02-19 12:51:36 EST to 13:05:19 EST (UTC: 17:51:36 to 18:05:19)

## 1) Environment context

- Repo: `/Users/bli/Playground/plato`
- `uv`: `0.9.18`
- Python: `3.13.11`
- Dependency probes:
  - `import lerobot` -> available
  - `import torch` -> available
  - `torch.cuda.is_available()` -> `False`

## 2) Commands executed and concrete outcomes

### A. Baseline preflight (lightweight, offline-safe)

Command:

```bash
/usr/bin/time -p uv run pytest -q \
  tests/test_config_loader.py::test_config_loads_smolvla_lerobot_parameter_contract \
  tests/datasources/test_lerobot_datasource.py \
  tests/models/test_smolvla_model.py \
  tests/trainers/test_lerobot_trainer.py \
  tests/integration/test_lerobot_smolvla_smoke.py \
  tests/algorithms/test_fedavg_algorithm.py
```

Outcome:

- Pass: `12 passed in 0.08s`
- Wall clock (`time -p`): `real 4.92`, `user 4.42`, `sys 0.50`

Interpretation:

- Local unit/integration coverage for LeRobot datasource, SmolVLA model wrapper, trainer, and FedAvg adapter behavior is healthy.

### B. Stage 1: Single-client local run

Command:

```bash
/usr/bin/time -p timeout 300 uv run python plato.py \
  --config configs/LeRobot/smolvla_single_client_smoke.toml
```

Observed key runtime behavior:

- Server and client initialized.
- LeRobot datasource loaded (`train episodes=165, test episodes=41`).
- Failure during round-1 model dispatch:
  - `TypeError: Got unsupported ScalarType BFloat16`
  - stack path includes `plato/processors/safetensor_encode.py` -> `plato/serialization/safetensor.py` -> `plato/utils/tree.py`.

Exit/timing:

- Fail: exit code `124` (timeout)
- `real 300.96`, `user 20.08`, `sys 15.71`

### C. Stage 2: 2-client federated smoke run

Command:

```bash
/usr/bin/time -p timeout 240 uv run python plato.py \
  --config configs/LeRobot/smolvla_fedavg_two_client_smoke.toml
```

Observed key runtime behavior:

- Server started with 2 clients configured.
- First client connected; round started.
- Failure on first payload send with the same exception:
  - `TypeError: Got unsupported ScalarType BFloat16`

Exit/timing:

- Fail: exit code `124` (timeout)
- `real 240.91`, `user 12.78`, `sys 3.63`

### D. Stage 3: Larger run (convergence/stability gate proxy)

Command:

```bash
/usr/bin/time -p timeout 120 uv run python plato.py \
  --config configs/LeRobot/smolvla_full_finetune.toml -u
```

Notes:

- `-u` used because this environment reports `torch.cuda.is_available() == False`.

Observed key runtime behavior:

- Server initialized (`Training: 10 rounds`).
- Datasource initialized (`train episodes=185, test episodes=21`).
- Failure at round-1 dispatch with the same exception:
  - `TypeError: Got unsupported ScalarType BFloat16`

Exit/timing:

- Fail: exit code `124` (timeout)
- `real 120.99`, `user 13.03`, `sys 4.22`

### E. Runtime artifact check

Command:

```bash
ls runtime/results | rg "^(94032|94157|94326)\\.csv$"
```

Observed files:

- `runtime/results/94032.csv`
- `runtime/results/94157.csv`
- `runtime/results/94326.csv`

Content check:

- Each file contains header only: `round,accuracy,elapsed_time`
- No completed round rows were recorded.

## 3) Baseline comparison

Expected baseline for staged gate:

- Single-client smoke: complete 1/1 round and exit without unhandled exception.
- Two-client smoke: complete 1/1 federated round with both clients selected and aggregated.
- Larger run: progress beyond round 1 to provide initial convergence/stability signal.

Actual:

- All three runs failed before completing round 1 due the same bfloat16 serialization issue.
- Therefore runtime behavior is below baseline for release readiness.

## 4) What could not be fully validated and why

- Full convergence/stability behavior (multi-round trend) could not be validated because execution stopped before any completed round.
- End-to-end federated completion for the 2-client path could not be validated for the same reason.
- CUDA path in `smolvla_full_finetune.toml` could not be validated in this environment (`torch.cuda.is_available() == False`).

## 5) Go/No-Go rollout gate

Decision: **NO-GO** for first public release in current state.

Blocking condition:

- Federated payload serialization does not currently handle bfloat16 tensors emitted by SmolVLA/LeRobot policy state, causing unhandled exceptions and hung runs until timeout.

Suggested release gate criteria to re-run after fix:

1. No unhandled exception across all three staged commands.
2. Single-client smoke completes within timeout and writes >=1 runtime CSV data row.
3. Two-client smoke completes round 1 with aggregation and writes >=1 runtime CSV data row.
4. Larger profile completes at least 3 rounds in the same environment class used for release qualification.

## 6) Recommended default settings for first public release

These are recommended defaults **after** the blocking serialization issue is fixed:

- Entry profile: `configs/LeRobot/smolvla_single_client_smoke.toml`
- `parameters.policy.finetune_mode = "adapter"`
- `parameters.policy.precision = "fp32"`
- `parameters.policy.device = "cpu"` for first-run smoke docs, then move to accelerator.
- `trainer.rounds = 1`, `trainer.epochs = 1`, `trainer.batch_size = 2` as onboarding default.
- Federated smoke default: `configs/LeRobot/smolvla_fedavg_two_client_smoke.toml` as second gate after single-client pass.

Operational note:

- Keep explicit timeout wrappers in CI/staging commands to avoid indefinite hangs when async server/client exceptions occur.
