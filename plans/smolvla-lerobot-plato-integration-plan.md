# SmolVLA + LeRobot Integration Plan for Plato

Date: 2026-02-19
Scope: Add support for training Hugging Face SmolVLA with LeRobot datasets/framework inside Plato.

## Dependency Graph

```text
T1 -> T2, T3
T2 -> T4, T5
T3 -> T4, T5, T9
T4, T5 -> T6
T6 -> T7, T8
T4, T5, T6 -> T9
T7, T8, T9 -> T10
T10 -> T11
T11 -> T12
```

## Tasks

### T1. Define integration contract and acceptance criteria
depends_on: []
status: completed (2026-02-19)
- Lock exact scope for first release:
- Support SmolVLA fine-tuning in Plato’s existing FL lifecycle.
- Support LeRobot dataset ingestion through Plato datasource APIs.
- Ensure compatibility with existing client/server/algorithm loops.
- Define acceptance checks:
- Single-client local training run works.
- Multi-client federated round-trip works.
- Config-first workflow (no code edits required to run experiment).
work_log:
- Added `plans/smolvla-lerobot-integration-contract.md` to lock v1 scope, concrete integration touchpoints, config contract, and testable acceptance checks.
- Made explicit scope boundaries and non-goals for v1.
files_touched:
- `plans/smolvla-lerobot-integration-contract.md` (created)
- `plans/smolvla-lerobot-plato-integration-plan.md` (updated)
gotchas:
- Context7 returned `LeRobot` as the primary documented surface; SmolVLA details were discovered through the LeRobot documentation set.

### T2. Add dependencies and environment gating
depends_on: [T1]
status: completed (2026-02-19)
- Update `pyproject.toml` with required LeRobot and training stack dependencies.
- Regenerate `uv.lock`.
- Add guarded imports so environments without robotics extras still run existing Plato workloads.
- Document required system/runtime notes for optional robotics path.
work_log:
- Added a new optional extra (`robotics`) in `pyproject.toml` with `lerobot[smolvla]>=0.4.3,<0.5.0` so default installs remain unchanged.
- Regenerated `uv.lock` with `uv lock`, then validated both paths:
- `uv sync --frozen` + import check for core Plato.
- `uv sync --frozen --extra robotics` + `import lerobot` check for the optional robotics stack.
- Added focused setup docs for SmolVLA/LeRobot and linked them from `docs/docs/install.md`.
files_touched:
- `pyproject.toml`
- `uv.lock`
- `docs/docs/install.md`
- `docs/docs/smolvla_lerobot_setup.md` (created)
- `plans/smolvla-lerobot-plato-integration-plan.md`
gotchas:
- The optional LeRobot path constrains parts of the Torch stack in lock resolution; keeping it under `--extra robotics` avoids forcing robotics dependencies into default `uv sync` environments.

### T3. Extend Plato configuration schema for SmolVLA/LeRobot
depends_on: [T1]
status: completed (2026-02-19)
- Add/validate config keys needed for SmolVLA + LeRobot:
- `policy.path` / `policy.type`
- `dataset.repo_id`
- `delta_timestamps`
- image transform controls
- precision/device flags
- full-finetune vs adapter mode switch
- Ensure keys flow through `Config()` and into trainer/model/datasource constructors.
work_log:
- Verified that `plato/config.py` already preserves nested TOML keys under `Config().parameters` without schema whitelisting, so SmolVLA/LeRobot keys are backward-compatible pass-through.
- Added a focused config loader test to assert `parameters.policy`, `parameters.dataset`, and `parameters.transforms` keys are parsed and exposed as constructor-ready dictionaries via `_asdict()`.
- Extended configuration documentation with an explicit mapping table from config keys to trainer/model/datasource consumption paths and a full SmolVLA/LeRobot TOML example.
files_touched:
- `tests/test_config_loader.py` (updated)
- `docs/docs/configurations/parameters.md` (updated)
- `plans/smolvla-lerobot-plato-integration-plan.md` (updated)
gotchas:
- No `Config()` code change was required; introducing strict validation at this stage would have been intrusive and risked regressions for existing custom `parameters.*` users.

### T4. Implement LeRobot datasource adapter
depends_on: [T2, T3]
status: completed (2026-02-19)
- Add `plato/datasources/lerobot.py`.
- Implement dataset loading via LeRobot APIs and map samples into Plato’s expected batch format.
- Register datasource in `plato/datasources/registry.py`.
- Add deterministic client partitioning strategy (episode/task aware split).
- Provide train/test dataset access methods compatible with existing samplers.
work_log:
- Added `plato/datasources/lerobot.py` with guarded LeRobot imports, config parsing for `parameters.dataset.*` and `parameters.transforms.*`, and sample mapping that preserves raw fields while attaching `plato_inputs`, `plato_targets`, and `plato_metadata`.
- Implemented deterministic episode-level train/test splitting with optional explicit episode overrides, task-aware stratification when task metadata is available, and deterministic per-client episode partitioning keyed by `data.random_seed`/`parameters.dataset.split_seed`.
- Wired `"LeRobot"` through `plato/datasources/registry.py` as a partitioned datasource so `datasources_registry.get(client_id=...)` passes client identity into the adapter.
- Ran a targeted no-download constructor/registry validation using stubbed `LeRobotDataset` and `LeRobotDatasetMetadata`, confirming deterministic splits and registry retrieval.
files_touched:
- `plato/datasources/lerobot.py` (created)
- `plato/datasources/registry.py` (updated)
- `plans/smolvla-lerobot-plato-integration-plan.md` (updated)
gotchas:
- Constructor/registry validation was intentionally monkeypatched to avoid external dataset access and to keep split checks deterministic and offline.
- When task metadata is sparse or missing, the adapter falls back to deterministic episode-only splitting.

### T5. Implement SmolVLA model/policy wrapper
depends_on: [T2, T3]
status: completed (2026-02-19)
- Add `plato/models/smolvla.py`.
- Implement pretrained loading path (`smolvla_base` and custom repo id/path).
- Expose trainable-parameter policy (full model or adapter path).
- Register model in `plato/models/registry.py`.
- Ensure state dict save/load compatibility with Plato aggregation pipeline.
work_log:
- Added `plato/models/smolvla.py` with lazy LeRobot import guards, actionable installation errors for missing robotics extras, and a SmolVLA factory path compatible with Plato model registry usage.
- Implemented pretrained policy source resolution with support for `smolvla_base` aliasing to `lerobot/smolvla_base`, config-based `parameters.policy.path`, and explicit constructor overrides (`policy_path` / `path`).
- Added finetune policy modes for `full` and `adapter`; adapter mode uses configurable name-pattern matching and falls back to the loaded policy's existing `requires_grad` flags when patterns do not match.
- Added compatibility checks for `state_dict`, `load_state_dict`, and `save_pretrained`, then registered `model_type = "smolvla"` in `plato/models/registry.py`.
- Ran a targeted constructor/import validation without downloads by monkeypatching `SmolVLAPolicy.from_pretrained` and verifying both direct wrapper construction and registry resolution.
files_touched:
- `plato/models/smolvla.py` (created)
- `plato/models/registry.py` (updated)
- `plans/smolvla-lerobot-plato-integration-plan.md` (updated)
gotchas:
- Adapter parameter names are model-dependent; when no configured adapter patterns match, the wrapper intentionally reuses the model's preconfigured trainable flags instead of silently leaving zero trainable tensors.

### T6. Implement LeRobot trainer backend
depends_on: [T4, T5]
status: completed (2026-02-19)
- Add `plato/trainers/lerobot.py` (ComposableTrainer-compatible).
- Implement multimodal collate + preprocessing for LeRobot samples.
- Wire forward/loss/backward/optimizer/scheduler flow for SmolVLA policy.
- Implement evaluation hooks suitable for regression checks.
- Register trainer in `plato/trainers/registry.py`.
work_log:
- Added `plato/trainers/lerobot.py` with a ComposableTrainer-compatible backend that wires custom dict/multimodal collation, processor-aware training steps, and evaluation loss reporting for regression checks.
- Implemented LeRobot pre/post-processor integration via `make_pre_post_processors(policy_cfg, pretrained_path=..., dataset_stats=...)`, with lazy optional-dependency import guards and actionable installation errors.
- Implemented SmolVLA policy forward integration handling tuple loss outputs and preserving optimizer + scheduler flow through the base composable lifecycle.
- Registered `trainer.type = "lerobot"` in `plato/trainers/registry.py`.
- Ran targeted offline validation with monkeypatched processor stubs:
- trainer registry resolution and construction (`trainer.type = "lerobot"`),
- synthetic one-epoch training-step path,
- synthetic evaluation pass returning numeric loss.
- Ran `uv run ruff check` on touched trainer files.
files_touched:
- `plato/trainers/lerobot.py` (created)
- `plato/trainers/registry.py` (updated)
- `plans/smolvla-lerobot-plato-integration-plan.md` (updated)
gotchas:
- LeRobot preprocessing depends on optional robotics extras at runtime; the trainer defers imports until processor initialization so non-robotics workloads remain unaffected, and it fails with a clear `uv sync --extra robotics` message when required dependencies are missing.

### T7. Harden federated payload/aggregation behavior
depends_on: [T6]
status: completed (2026-02-19)
- Ensure only intended trainable tensors are exchanged/aggregated.
- Add safeguards for payload size and dtype handling.
- Verify checkpoint/state restore consistency across rounds.
- Validate no regressions in FedAvg flow with large model weights.
work_log:
- Hardened `plato/algorithms/fedavg.py` to exchange adapter-only tensors when `plato_finetune_mode = "adapter"` and `plato_trainable_parameter_names` are provided, while preserving full-state behavior for existing non-adapter models.
- Added dtype-safe tensor casting and partial payload merge logic in `load_weights()`, plus stricter key/shape validation and delta application safeguards for partial/full state dicts across rounds.
- Added payload-size safeguards with an optional limit (`model.plato_max_payload_size_mb` or `PLATO_FEDAVG_MAX_PAYLOAD_MB`) and fail-fast checks when payloads exceed the configured cap.
- Added targeted regression tests for filtered extract/load round-trip, dtype safety, optional payload-size guard, and full-mode FedAvg round-trip with large weights.
- Ran `uv run ruff check` and focused `uv run pytest` for the new FedAvg algorithm tests.
files_touched:
- `plato/algorithms/fedavg.py` (updated)
- `tests/algorithms/test_fedavg_algorithm.py` (created)
- `plans/smolvla-lerobot-plato-integration-plan.md` (updated)
gotchas:
- Payload-size enforcement is intentionally opt-in to maintain backward compatibility for existing workloads that may exchange large full-model state dicts.

### T8. Validate runtime lifecycle compatibility
depends_on: [T6]
status: completed (2026-02-19)
- Confirm integration works with existing lifecycle code paths:
- client setup strategies
- server trainer initialization
- training/report/aggregation loop
- Avoid special-case branching unless strictly necessary.
work_log:
- Ran a focused runtime smoke with `data.datasource = "LeRobot"`, `trainer.type = "lerobot"`, and `trainer.model_type = "smolvla"` using monkeypatched LeRobot/SmolVLA externals to avoid downloads, then exercised the default `simple.Client` lifecycle (`_load_data` -> `configure` -> `_allocate_data` -> `_train`).
- Verified lifecycle construction path through existing registries and strategy plumbing: datasource (`LeRobot`) + trainer (`lerobot`) + algorithm (`fedavg`) were all instantiated through default client/server setup with no special-case branching.
- Executed a short mocked client/server round-trip by feeding the client-produced payload/report into `fedavg.Server._process_reports()` after `Server.configure()`, confirming server trainer initialization and aggregation/report processing completed successfully.
- No lifecycle compatibility bug was found in this scope, so no runtime code patch was applied.
files_touched:
- `plans/smolvla-lerobot-plato-integration-plan.md` (updated)
gotchas:
- The focused smoke directly called `client._train()`; because `report.processing_time` is normally attached in the payload strategy path, the smoke sets `report.processing_time = 0.0` before invoking server report processing.

### T9. Add runnable experiment configs
depends_on: [T3, T4, T5, T6]
status: completed (2026-02-19)
- Add `configs/LeRobot/` config set:
- reusable base datasource fragment
- minimal smoke config
- full fine-tune config aligned to SmolVLA guidance
- Ensure includes/overrides follow repository config conventions.
work_log:
- Added `configs/LeRobot/` with a reusable datasource include fragment plus runnable single-client smoke, two-client FedAvg smoke, and fuller full-fine-tune configs.
- Aligned all new configs with T4-T6 integration keys: `data.datasource = "LeRobot"`, `trainer.type = "lerobot"`, `trainer.model_type = "smolvla"`, and explicit `[parameters.policy]`, `[parameters.dataset]`, `[parameters.transforms]` sections.
- Mapped SmolVLA fine-tuning guidance into Plato semantics by keeping `policy.path = "lerobot/smolvla_base"`, `policy.finetune_mode = "full"`, `policy.device = "cuda"`, and `batch_size = 64` in the fuller config.
files_touched:
- `configs/LeRobot/lerobot_datasource_base.toml` (created)
- `configs/LeRobot/smolvla_single_client_smoke.toml` (created)
- `configs/LeRobot/smolvla_fedavg_two_client_smoke.toml` (created)
- `configs/LeRobot/smolvla_full_finetune.toml` (created)
- `plans/smolvla-lerobot-plato-integration-plan.md` (updated)
gotchas:
- The datasource include fragment is intentionally sectionless so `[data].include` merges it directly into the `data` table.
- SmolVLA upstream examples are step-based (`lerobot-train --steps`), while Plato scheduling is round/epoch-based, so the fuller config mirrors guidance through batch/device/fine-tune mode and keeps runtime knobs in `trainer.rounds` + `trainer.epochs`.

### T10. Add tests (unit + integration smoke)
depends_on: [T7, T8, T9]
status: completed (2026-02-19)
- Datasource registry + constructor tests for LeRobot datasource.
- Model registry + construction tests for SmolVLA wrapper.
- Trainer step test with tiny synthetic batch.
- End-to-end config smoke test covering startup and one short training run.
- Add regression tests for any bug fixes discovered during integration.
work_log:
- Added focused LeRobot datasource tests covering partitioned registry resolution, deterministic constructor split behavior, and mapped `plato_inputs`/`plato_targets` sample keys.
- Added SmolVLA model tests covering registry-based wrapper construction and a FedAvg regression check asserting adapter-mode metadata results in adapter-only payload extraction.
- Added a LeRobot trainer tiny-batch unit test that exercises one short training step with synthetic dict samples, stubbed pre/post processors, and parameter-update assertions.
- Added an end-to-end LeRobot+SmolVLA smoke test that boots from config, runs one short client training pass, and processes a server FedAvg report/update loop with external dependencies fully monkeypatched.
- Fixed local test import shadowing discovered during validation by adding package markers under `tests/` and `tests/test_utils/`.
files_touched:
- `tests/__init__.py` (created)
- `tests/test_utils/__init__.py` (created)
- `tests/test_utils/lerobot_stubs.py` (created)
- `tests/datasources/test_lerobot_datasource.py` (created)
- `tests/models/test_smolvla_model.py` (created)
- `tests/trainers/test_lerobot_trainer.py` (created)
- `tests/integration/test_lerobot_smolvla_smoke.py` (created)
- `plans/smolvla-lerobot-plato-integration-plan.md` (updated)
gotchas:
- The local environment includes a third-party `tests` package in site-packages; without `tests/__init__.py`, pytest imports can resolve to the wrong module namespace.

### T11. Add documentation and runbook
depends_on: [T10]
status: completed (2026-02-19)
- Document setup and dependency extras.
- Document config fields and examples.
- Add troubleshooting notes (dataset access, device setup, common failures).
- Add mapping between Plato config and equivalent `lerobot-train` concepts.
work_log:
- Added an operator-facing runbook covering dependency setup, runnable commands, minimum TOML contract, and troubleshooting for common LeRobot/SmolVLA failures.
- Added an explicit Plato TOML to `lerobot-train` mapping table with direct flag mappings (`policy.path`, `dataset.repo_id`, `batch_size`, `policy.device`) and conceptual mappings (`rounds`/`epochs` vs `steps`, output paths).
- Referenced all new `configs/LeRobot/*` profiles directly in the runbook and linked the runbook from installation docs and top-level docs navigation.
- Grounded mapping/troubleshooting notes against current LeRobot documentation via Context7 and implementation-specific runtime errors from Plato's LeRobot datasource/trainer/model integration.
files_touched:
- `docs/docs/smolvla_lerobot_runbook.md` (created)
- `docs/docs/install.md` (updated)
- `docs/mkdocs.yml` (updated)
- `plans/smolvla-lerobot-plato-integration-plan.md` (updated)
gotchas:
- `lerobot-train` examples are primarily step-based (`--steps`) while Plato scheduling is round/epoch-based; documentation uses explicit conceptual mapping instead of implying a one-to-one flag conversion.

### T12. Stage validation and rollout gate
depends_on: [T11]
- Execute staged validation:
- single-client local run
- 2-client federated smoke run
- larger run for convergence and stability check
- Compare behavior/runtime against expected baseline.
- Define go/no-go criteria and recommended defaults for first public release.

## Milestones

- Milestone A (Core plumbing): T1-T6 complete.
- Milestone B (Federated reliability): T7-T8 complete.
- Milestone C (Usability + confidence): T9-T12 complete.

## Notes

- Existing codebase discovery found no native SmolVLA/LeRobot implementation yet.
- Primary extension anchors are current registries and Hugging Face integration patterns.
