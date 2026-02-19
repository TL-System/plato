# SmolVLA + LeRobot Integration Contract (Release v1)

Date: 2026-02-19  
Plan Task: T1 (`depends_on: []`)  
Status: accepted baseline for implementation tasks T2+

## 1. Objective

Define a concrete, testable contract for first-release SmolVLA + LeRobot support in Plato without changing Plato's core federated runtime model.

## 2. In-Scope (Release v1)

1. SmolVLA fine-tuning runs inside Plato's existing federated lifecycle.
2. LeRobot datasets are ingested via Plato datasource APIs.
3. Existing client/server/algorithm loops remain the orchestration path.
4. End users run experiments through TOML configuration only (no source edits).

## 3. Integration Surface (Concrete Components)

The implementation must integrate through these existing extension points.

Runtime entry and lifecycle:
1. `plato.py`
2. `plato/client.py`
3. `plato/clients/registry.py`
4. `plato/clients/base.py`
5. `plato/servers/fedavg.py`
6. `plato/servers/registry.py`
7. `plato/algorithms/registry.py`

Config loading and propagation:
1. `plato/config.py`

Datasource extension points:
1. `plato/datasources/base.py`
2. `plato/datasources/registry.py`
3. New module target: `plato/datasources/lerobot.py`

Model extension points:
1. `plato/models/registry.py`
2. New module target: `plato/models/smolvla.py`

Trainer extension points:
1. `plato/trainers/base.py`
2. `plato/trainers/composable.py`
3. `plato/trainers/registry.py`
4. New module target: `plato/trainers/lerobot.py`

Compatibility rule: v1 must work with existing `fedavg` server/algorithm paths. Any special handling must be encapsulated inside the new datasource/model/trainer modules and their registry wiring.

## 4. Configuration Contract (v1)

The following fields are the required contract for SmolVLA + LeRobot configs. T3 is responsible for schema wiring/validation.

```toml
[data]
datasource = "LeRobot"
# existing partitioning keys stay valid (sampler, partition_size, random_seed)

[trainer]
type = "lerobot"
model_type = "smolvla"
model_name = "smolvla"

[parameters.policy]
type = "smolvla"
path = "lerobot/smolvla_base"
finetune_mode = "full" # "full" or "adapter"
precision = "bf16"     # expected values: fp32/fp16/bf16
device = "cuda"         # expected values: cpu/cuda/mps

[parameters.dataset]
repo_id = "<org-or-user>/<dataset>"
delta_timestamps = { observation_image = [-0.2, -0.1, 0.0] }

[parameters.transforms]
image_size = [224, 224]
normalize = true
```

Required semantics:

1. `parameters.policy.path` resolves pretrained policy source.
2. `parameters.policy.type` selects policy family (`smolvla` for v1).
3. `parameters.dataset.repo_id` selects LeRobot dataset source.
4. `parameters.dataset.delta_timestamps` controls temporal windowing.
5. `parameters.transforms.*` controls image preprocessing.
6. `parameters.policy.finetune_mode` controls full vs adapter updates.
7. `parameters.policy.precision` and `parameters.policy.device` govern runtime dtype/device behavior.

## 5. Scope Boundaries and Non-Goals

Explicitly out of scope for release v1:

1. New federated algorithms or server types beyond existing registry options.
2. Live robot inference/control loops and async teleoperation workflows.
3. Non-LeRobot robotics dataset backends.
4. End-to-end convergence/benchmark claims beyond smoke/stability checks.
5. Automated dependency bootstrap for platform-specific robotics stacks.

## 6. Acceptance Checks (Concrete and Testable)

These checks define go/no-go for the integration scope.

### AC1: Single-client local training run

- Config target: `configs/LeRobot/smolvla_single_client_smoke.toml`.
- Command:

```bash
uv run python plato.py --config configs/LeRobot/smolvla_single_client_smoke.toml
```

Pass criteria:
1. Process exits with code `0`.
2. Trainer completes at least one local epoch in one communication round.
3. A model artifact is written under configured `model_path`.

### AC2: Multi-client federated round-trip

- Config target: `configs/LeRobot/smolvla_fedavg_two_client_smoke.toml`.
- Command:

```bash
uv run python plato.py --config configs/LeRobot/smolvla_fedavg_two_client_smoke.toml
```

Pass criteria:
1. Server starts and selects two clients in the same round.
2. Server receives two client updates and completes aggregation.
3. Round counter advances to at least round 1 completion without runtime exceptions.

### AC3: Config-first workflow (no source edits)

Validation procedure:
1. Run `AC1` and `AC2` using committed TOML files only.
2. Confirm no local source modifications are required between runs.

Pass criteria:
1. Both runs succeed from clean checkout with only config selection changed.

## 7. Deliverables Expected From Downstream Tasks

1. Config files under `configs/LeRobot/` implementing AC1/AC2 targets.
2. Registry wiring and implementation modules listed in Section 3.
3. Smoke/integration tests that encode AC1/AC2/AC3 behavior.

## 8. Notes

- This contract intentionally locks only v1 integration behavior and acceptance gates.
- Performance tuning and broader robotics feature surface are deferred to post-v1 tasks.
