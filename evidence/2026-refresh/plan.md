# Plato 2026 refresh: implementation plan and audit inventory

Status: proposed, awaiting independent plan review before coding.

- Planning date: 2026-10-02.
- Root branch: `maintenance/2026-refresh-audit`.
- Exact baseline commit: `5b8359bd40bbe65c7753e71f6627c490f9aa144f`.
- This document is the planning agent's sole write ownership. The planner makes
  no implementation edits, commits, or nested agents.
- Root owns task assignment, integration, commits, push, PRs, and final cleanup.
- Repository guidance: `AGENTS.md`; all shell commands use `zsh -lc`.
- Required workflow: <https://baochun.org/2026-09-05/>.

## 1. Scope and decisions

Refresh dependencies and development tooling against current authoritative
metadata; make installation reproducible; qualify Python 3.14; audit all runtime
components, research examples, documentation, tests, and delivery tooling; repair
demonstrated defects without gratuitous refactoring.

The following boundaries are fixed for this plan:

1. Keep Python 3.13 as the minimum supported version. Qualify Python 3.14.8 as
   the default only after fresh-process startup and runtime checks pass. A
   successful resolver or unit-suite run alone does not establish support.
2. Python.org currently lists 3.14.8 as stable and 3.15 as prerelease. Do not
   target 3.15 or free-threaded Python in this refresh.
3. Preserve research algorithms, configuration meaning, update equations, and
   checkpoint expectations. Reproduce suspected defects before changing them.
   Research semantic rewrites require user approval.
4. Audit and repair transport implementations within their existing contract.
   Socket/S3 paths that deserialize pickle are trusted-peer-only; payload checks
   do not make pickle safe for untrusted clients or object writers. Authentication
   and transport protocol redesign require user approval as separate roadmap
   work. Public-deployment documentation must state this trust boundary.
5. Repair and validate the existing MLX path. Larger MLX features are a proposed
   roadmap, not authorized implementation scope.
6. Preserve archived examples and third-party attribution. Restoring an archive
   to supported status, deleting it, or changing upstream submodule versions for
   reasons beyond a demonstrated compatibility need requires a scope decision.
7. Do not replace useful tests with weaker assertions, hide required coverage
   behind skips, broadly suppress typing diagnostics, or add tests for mechanical
   edits that merely duplicate implementation details.

## 2. Evidence and current baseline

Repository inspection covered the tracked-file inventory, Python syntax/import
structure, TOML parsing, manifests, workflows, documentation configuration, and
targeted implementation/test paths. This is a planning audit, not a claim that
every runtime path has been behaviorally validated.

| Evidence | Current result | Consequence |
| --- | --- | --- |
| Git baseline | Exact commit above; no tracked implementation edits reported | Review and compare changes against this baseline |
| Package manager | `uv 0.12.22` available under `zsh -lc` | Earlier observation that uv was unavailable is superseded |
| Workspace/install | `uv sync` succeeds; nonexistent `tools` member is silently tolerated | Stale metadata, not a demonstrated install blocker |
| Lockfile | Root generated a local lock; baseline does not track it and `.gitignore` excludes `uv.lock` | Reproducibility remains an implementation task |
| Python 3.13 | 3.13.16 baseline with `dp/mpc/rl/nanochat`: 248 passed, 3 skipped, 8 multiprocessing fork warnings | Record exact commands and skip/warning reasons; preserve useful coverage |
| Base collection | Without the optional DP extra, collection fails on missing Opacus | Define base and mandatory optional test profiles explicitly |
| Python 3.14 | 3.14.8 installed; isolated environment `/tmp/plato-python314-env` installed and suite passed | Installation and suite feasibility established; runtime support is still blocked |
| Python 3.14 startup | Root's fresh-process `client.run()` probe raises `RuntimeError` because no event loop exists | Mandatory Phase 1 bootstrap fix and subprocess regressions |
| Startup sites | `plato/client.py:89`; startup calls in `plato/servers/base.py:358,372,407` at baseline | Inspect loop ownership across client/server paths before choosing a fix |
| MLX | Root's real LeNet train/test/save/delta probes passed | Existing path has positive execution evidence; expand regression coverage rather than describe it as wholly untested |
| MLX malformed trees | Root confirms unequal lists truncate, mismatched shapes broadcast, and extra keys are ignored | Confirmed F17 requires explicit rejection tests and repair |
| Coverage | Root reports 43% baseline coverage | Preserve exact invocation/report; use it to locate behavioral gaps, not impose an arbitrary percentage target |
| Vulnerabilities | Root reports installed dependency vulnerability audit clear | Preserve tool/version/environment/date; this does not establish coverage for every optional profile or future lock |
| Ruff | Import checks pass | Keep passing; no unrelated formatting churn |
| Typing | 59 diagnostics reported | Classify behavior defects, interface issues, optional imports, and research/vendor typing separately |
| Documentation | Root reports two broken relative links from MkDocs checks | Capture exact pages/targets and repair during docs work |
| GitHub | Root confirms connected GitHub app has push/admin access and PR creation tools; SSH git origin is reachable | Publication blocker resolved; `gh` authentication is unnecessary for the available workflow |
| Submodules | Root confirms Nanochat, DViT, T2T-ViT now initialized at pinned SHAs | Initial missing-submodule observation is superseded; validate integrations against those pins |
| Docker | Unavailable in the current environment | Container execution needs an available runner/host; static inspection is not a passing build/runtime check |
| Independent Sol baseline audit | Root reports all defects in the ledger below confirmed by probes | Explicit repair tasks, not speculative audit candidates |

Root-provided execution results above were incorporated into the plan; the
planning agent has not independently rerun every reported probe. Root should
retain their original logs, exact commands, environment identities, and dates.

Root-owned durable evidence now present alongside this plan:
[package metadata](package-metadata.json),
[baseline coverage](baseline-coverage.json), and
[baseline vulnerability audit](baseline-vulnerabilities.json).
These files are outside the planning agent's write ownership. Prefer these
retained artifacts over temporary copies when assembling review evidence.

### Confirmed defect ledger and acceptance tasks

All rows below are **confirmed and open**, based on root's probe evidence and
independent Sol baseline audit. Root must retain the original probes and convert
them into focused red/green regressions. Each row is an explicit acceptance task
within its named owner, not a discretionary inspection candidate. A row closes
only with regression evidence, independent review, and integration validation.

| ID | Confirmed defect | Phase owner and exact implementation scope | Required acceptance |
| --- | --- | --- | --- |
| F01 | Python 3.14 synchronous startup has no current event loop | 1B: `plato/client.py`, `plato/servers/base.py`; fresh-process integration tests | Actual client/server entrypoints execute scheduled work on 3.13 and 3.14 without a fixture-created loop; exceptions propagate and shutdown is bounded; real minimal round passes |
| F02 | S3 keys receive the prefix twice | 2D: `plato/utils/s3.py`; focused S3 utility tests | Configured prefix is applied exactly once across put/send/receive/delete/list workflows; empty-prefix and prefixed-bucket cases round-trip using the same object key |
| F03 | S3 empty listings fail | 2D: `plato/utils/s3.py`; same test owner as F02 | A valid empty response with no `Contents` produces an empty result, while genuine service errors remain visible |
| F04 | S3 listing omits later pages | 2D: `plato/utils/s3.py`; same test owner as F02 | Multiple pages are consumed completely, respecting the configured namespace, with no dropped or duplicated keys |
| F05 | S3 network operations lack bounded timeouts | 2D: `plato/utils/s3.py`; same test owner as F02 | Blocking request paths use finite connect/read timeout and retry behavior; deterministic stalled/error probes terminate with useful failures, while successful transfers still work |
| F06 | All three LeRobot run configs select IID without required `partition_size` | 3B: `configs/LeRobot/lerobot_datasource_base.toml`, `smolvla_single_client_smoke.toml`, `smolvla_fedavg_two_client_smoke.toml`, `smolvla_full_finetune.toml`; config-driven integration tests | Load each real TOML/include chain and construct its actual sampler against a bounded dataset; valid nonempty partitions and sample counts result. Preserve intended dataset/client partition semantics; ask before changing research partition policy |
| F07 | Slash-containing model names break worker/default/server-resume checkpoint paths | 2C lead: `plato/trainers/base.py`, `plato/trainers/composable.py`; root hands off checkpoint paths in `plato/clients/base.py` and `plato/servers/base.py` from 2B | A name such as `org/model` works for default save/load, worker train/test artifacts and server checkpoint/resume; writers/readers agree, auxiliary files survive round trips, paths remain within configured roots, and ordinary existing names stay compatible |
| F08 | Generic gradient accumulation drops the tail and omits `optimizer_step_completed` | 2C: `plato/trainers/strategies/training_step.py`, integration points in `plato/trainers/composable.py`; trainer strategy tests | Non-divisible and shorter-than-window inputs flush exactly one correctly normalized final update; divisible inputs add none; step flag is accurate on every microbatch/finalize; optimizer/scheduler/callback counts match actual updates and gradients do not leak across runs |
| F09 | HuggingFace partial accumulation window is undernormalized | 3B: `plato/trainers/huggingface.py`, `tests/trainers/test_huggingface_trainer.py`; depends on F08 contract | Deterministic numerical gradients/parameter updates for a partial window match the intended unaccumulated reference; full windows remain correct; finalize does not double-step and callback/scheduler counts remain accurate |
| F10 | Disconnect cleanup deletes other clients' checkpoints | 2B: `plato/clients/base.py` cleanup and relevant disconnect call sites; lifecycle tests | Disconnect one client with two clients' artifacts present: remove only that client's owned temporary files; preserve the other client's checkpoints, server resume state, and unrelated files |
| F11 | PORT expects `.pth` checkpoints incompatible with the Safetensors trainer | 2B: `plato/servers/strategies/aggregation/port.py` reader plus limited handoff of `examples/async/port/port_server.py` writer and focused `tests/servers/test_port_checkpoint_flow.py`; coordinate with 2C/F07 | Invoke the real `weights_aggregated` hook to write the checkpoint, then historical retrieval and stale-similarity computation; verify against a numerical reference with no format/extension mismatch. Preserve historical checkpoint compatibility where already promised; do not redesign the checkpoint protocol |
| F12 | Dictionary paths and reserved metadata keys collide in tree serialization | 2D: `plato/utils/tree.py`, `plato/serialization/safetensor.py`, `tests/utils/test_safetensor_serialization.py` | Dotted/bracketed keys versus nested paths and reserved metadata keys cannot silently overwrite data; round-trip losslessly within the existing format or reject ambiguous input explicitly before transmission. Ordinary existing payloads remain compatible; a format/protocol redesign requires approval |
| F13 | Config optional sections survive singleton reset | 2A: `plato/config.py`, `tests/test_config_loader.py` | Load config A with optional sections, reset, then load B without them: B exposes only its own data/defaults; reverse/repeated order is isolated without breaking documented CLI overrides |
| F14 | Timm first update is numbered 11 instead of 1 with ten batches | 2C: `plato/trainers/strategies/lr_scheduler.py`, related tests | First epoch updates are numbered 1–10; subsequent epochs/rounds and resume/global-LR offsets follow the intended sequence; accumulated microbatches advance only on real optimizer updates |
| F15 | Dirichlet `num_samples()` redraws the sample size | 2A: `plato/samplers/dirichlet.py`, sampler tests | Repeated count queries agree with the realized partition/loader size and do not consume RNG or mutate sampling; seeded construction is reproducible and client counts remain accurate |
| F16 | IID sampling hangs on an empty dataset | 2A: `plato/samplers/iid.py`, sampler tests | Empty data terminates promptly with the defined error/empty behavior in a timeout-guarded process; nonempty padding/partition behavior stays unchanged |
| F17 | MLX malformed parameter trees silently truncate unequal lists, broadcast mismatched shapes, and ignore extra keys | 3A: `plato/algorithms/mlx_fedavg.py`, relevant validation in `plato/trainers/mlx.py`; focused MLX algorithm tests | Reject unequal sequence lengths, missing/extra dictionary keys and mismatched leaf shapes before subtraction/addition or model mutation; test both broadcast-compatible and incompatible shape mismatches, unchanged state after failure, and valid-tree numerical behavior |

Dependency constraints within the ledger: F01 precedes Python 3.14
qualification; F07 coordinates all checkpoint writers/readers before F11 is
accepted; F08 establishes the shared optimizer-step contract before F09 and F14
integration; 2A's sampler contracts precede F06 qualification; F12 precedes final
MLX/external transport qualification; F17 precedes MLX supported-path acceptance.
Root serializes shared-file tasks such as
F07/F08 and F02–F05 rather than assigning concurrent conflicting edits.

Further findings, such as REFER filtering and optional LoRA callability, remain
separate audit work until their exact behavior is established. Their candidate
status must not be conflated with the confirmed rows above.

Authoritative checks made during planning:

- Python releases: <https://www.python.org/downloads/>.
- Python multiprocessing changes:
  <https://docs.python.org/3/library/multiprocessing.html>.
- uv workspace semantics:
  <https://docs.astral.sh/uv/concepts/projects/workspaces/>.
- Package metadata: `https://pypi.org/pypi/<distribution>/json`; root's 49-record
  snapshot is currently `/tmp/plato-package-metadata.json`.
- MLX 0.32.3 documentation:
  <https://ml-explore.github.io/mlx/build/html/usage/saving_and_loading.html>.
- GitHub Actions and container-tooling releases: authoritative upstream
  repositories, with compatibility notes reviewed before changing references.

## 3. Review, task ownership, and completion rules

Before coding, root obtains a fresh independent Sol xhigh review of this plan,
its ownership boundaries, dependencies, acceptance criteria, and evidence. Root
resolves substantive findings and approves the task breakdown. The planner does
not spawn that reviewer.

An earlier revision received independent Sol review findings. This revision
incorporates the accepted security, real-runtime-smoke, MLX and example-gate
changes; root must obtain a fresh review of this exact file hash before coding.

For execution, follow the linked workflow and the user's explicit steering:

1. Root starts each implementation task in an independent herdr workspace.
   Use `gpt-6.1-sol` at xhigh for code and independent task reviews, Astra high
   for writing/planning, and Astra medium for each phase gate.
2. No agent starts nested agents. Root alone orchestrates workers and reviewers.
3. Each task below is an ownership envelope. Before assignment, root narrows
   each demonstrated defect into a bounded task with exact files, prerequisites,
   and acceptance criteria. Inspection does not authorize a broad rewrite.
4. Root serializes shared changes to manifests, lockfiles, registries, fixtures,
   workflows, and shared serialization helpers. Transfer ownership explicitly;
   do not allow concurrent edits to the same file.
5. A substantive fix needs a failing behavioral reproduction before the fix and
   a passing regression afterward. Mechanical documentation/metadata edits use
   appropriate build, resolution, link, or command validation instead.
6. A fresh independent reviewer checks each task. Reuse the implementer to fix
   accepted critical/high/medium findings, then obtain a fresh review. Do not
   weaken acceptance criteria to obtain approval.
7. Review evidence identifies the exact base commit and reviewed diff identity;
   root-created commits receive exact-commit verification. Revalidate when
   integration changes the reviewed implementation.
8. Each phase requires its Astra medium gate before root commits the phase and
   proceeds. Root records acceptance, integrates, validates the merged tree,
   pushes, updates the PR, and only then cleans task-owned resources. Keep the
   original implementer available if integration fails.
9. Root opens the PR after Phase 1, then updates it after each phase. The
   connected GitHub app and reachable SSH origin provide the authorized remote
   path; do not impose an unnecessary `gh auth` prerequisite.

Completion evidence belongs under `evidence/2026-refresh/` in root-assigned
files other than this planner-owned document. Retain essential logs and concise
machine-readable inventories; remove only task-owned disposable artifacts.

## 4. Phase 1 — Dependencies, startup compatibility, and baseline

Dependency order: **1A -> 1B -> 1C -> independent phase gate**. No default Python
upgrade or claim of Python 3.14 support before the gate.

### 1A. Reproducible installation and dependency decisions

Ownership: root `pyproject.toml`, all 15 example `pyproject.toml` files,
`.gitignore`, new tracked `uv.lock`, installation-tool settings, and dependency
evidence. Reserve `.python-version` default promotion for 1C.

Work:

- Review all declared versions against current metadata, then inspect used APIs,
  transitive constraints, wheel availability, platforms, and release notes.
- Record current resolution, candidate, source/date, compatibility evidence,
  selected version/range, and keep/upgrade/defer rationale for every entry.
- Remove the stale `tools` workspace entry after confirming no intended package
  is being omitted; do not manufacture a placeholder package.
- Reconcile missing README declarations and workspace dependency relationships.
  Preserve example environments that require incompatible research stacks.
- Track a reviewed lock; define base, development/test, and optional profiles.
  Keep Python minimum 3.13 and validate both 3.13 and 3.14 resolution.
- Check source and built-artifact installation outside the checkout, including
  import behavior when optional extras or submodules are absent.

Acceptance:

- A fresh checkout can install with documented `uv sync --locked` profiles.
- `uv lock --check` succeeds without changing the lock.
- Isolated wheel/sdist builds and artifact installation/import smoke pass.
- Every dependency has a recorded decision; no blanket upgrade justification.
- The selected test profile includes Opacus where DP tests are mandatory.
- Installed vulnerability evidence is retained; rescan changed resolved profiles
  without treating the current clear result as permanent.

### 1B. Python 3.14 event-loop bootstrap fix

Ownership: `plato/client.py`, startup code in `plato/servers/base.py`, focused new
tests such as `tests/integration/test_event_loop_startup.py`, and test-only
subprocess helpers. One task owner coordinates both runtime files.

Work:

- Capture the existing fresh-process Python 3.14 failure at `client.run()`.
- Trace the server startup sites, edge/custom-client branches, task scheduling,
  loop sharing, exception propagation, and shutdown ownership.
- Make explicit loop lifecycle changes that preserve existing runtime behavior.
  Do not blindly replace all calls with `asyncio.run()` or mask the failure by
  creating a loop in a fixture before entrypoint invocation.
- Check existing explicit spawn paths and implicit multiprocessing pools;
  preserve a compatible process context rather than globally forcing a new one.

Acceptance:

- Red tests launch a fresh target interpreter and reproduce the client defect
  and any independently confirmed server defects on the unchanged baseline.
- Green tests pass on Python 3.13.16 and 3.14.8 through the actual synchronous
  entrypoints, with no precreated loop or mocked asyncio lifecycle.
- Cover ordinary startup and materially different edge/custom-client paths;
  assert that scheduled work actually executes and failures reach the caller.
- Shutdown is bounded, children exit, and no task/process leak is hidden by a
  successful parent exit. Use test timeouts and explicit process cleanup.
- A small real client/server round passes on both Python versions with
  `comm_simulation = false`, actual socket payload exchange through the existing
  processors/serialization, and independently trained clients whose parameters
  demonstrably change. Compare the server aggregate against a numerical
  sample-weighted reference from the actual received client updates; do not
  substitute fabricated reports/deltas or an in-process aggregation call.
- The real smoke uses isolated ports/runtime paths, a deadline, and child-process
  tracking. Assert normal shutdown and bounded cleanup on failure/timeout, with
  no surviving task-owned children. Lightweight bootstrap probes remain useful
  regressions but do not replace this smoke.

### 1C. Baseline qualification and minimal CI enforcement

Ownership: `.python-version`, test configuration in `pyproject.toml` after 1A
handoff, `tests/conftest.py`, `tests/integration/utils.py`, minimal changes to
`.github/workflows/pytorch_tests.yml`, baseline/coverage evidence.

Work and acceptance:

- Run base collection and required optional profiles in fresh environments.
  Optional tests may skip only under an explicitly optional profile; the job
  claiming support for an extra must execute its tests.
- Record exact commands, environment/lock identity, counts, skips, warnings,
  duration, Ruff results, typing diagnostics, and the 43% coverage baseline.
- Compare Python 3.13 and 3.14 suites plus the startup/runtime checks from 1B.
  Triage the eight fork warnings; do not suppress them as a substitute for
  investigating process-context behavior.
- Classify all 59 typing diagnostics and assign actionable findings to later
  owners. Typing-only cleanup must not change research behavior unnecessarily.
- Establish a bounded CPU runtime smoke in CI, not just unit tests.
- Promote the default to 3.14.8 only if qualification passes; otherwise keep
  3.13 as default and record the exact blocker. Minimum remains 3.13 either way.

Phase gate: reproducible install, clean collection for declared profiles,
startup regressions, actual runtime smoke, and honest baseline evidence are all
required. Root completes commit/push/PR milestones before Phase 2.

## 5. Phase 2 — Core behavior and component audit

Prerequisite: accepted Phase 1. Tasks may inspect disjoint areas in parallel;
root orders fixes crossing component boundaries and owns shared-file handoffs.
Exclude MLX and named external adapters reserved for Phase 3.

### 2A. Configuration, datasources, and samplers

Ownership: `plato/config.py`, non-adapter `plato/datasources/**`,
`plato/samplers/**`, `tests/test_config_loader.py`, relevant datasource/sampler
tests. Exclude HuggingFace, LoRA, LeRobot, Nanochat adapters reserved for 3B.

Audit includes/overrides, config singleton lifecycle, datasource aliases,
partitioning, seed isolation, empty data, download/extraction errors, and datalib
utilities. Reproduce the REFER image-filter nested-list behavior and discarded
annotation intersection; reconcile root's reported REFER typing finding against
actual behavior. Fix confirmed F13 (Config reset), F15 (Dirichlet counts), and
F16 (empty IID hang) with the explicit ledger acceptance tests; the bounded IID
reproduction is already confirmed and must become a regression.

Acceptance: correct IDs/partitions, deterministic sampling, bounded meaningful
errors on invalid inputs, preserved valid config semantics, and negative-path
regressions for confirmed defects. Do not redesign dataset APIs.
Archive extraction paths also satisfy the concrete containment checks in 2D,
with 2A owning datasource implementations and tests.

### 2B. Runtime lifecycle and strategy contracts

Ownership: `plato.py`, `plato/client.py` and `plato/servers/base.py` after 1B
handoff, `plato/clients/**`, `plato/servers/**`, `plato/callbacks/**`, matching
tests. Coordinate specialized MPC/HE paths with 2D rather than editing them
concurrently.

Audit legacy hooks versus composable strategies, context synchronization,
payload/report alignment, aggregation round boundaries, client selection,
disconnects, cancellation, evaluation logging, and sync/async/buffered/split/
cross-silo lifecycles.

Acceptance: numerical aggregation/selection invariants, correct round progress,
surfaced child errors, bounded teardown, and a representative real orchestration
smoke. Socket identity/size checks also satisfy 2D's security acceptance, with
2B owning socket handlers and lifecycle tests. Complete confirmed F10
(disconnect cleanup) and F11 (PORT checkpoints: limited Phase 2 ownership of `examples/async/port/port_server.py` and its focused hook-to-retrieval regression alongside the strategy reader),
and hand checkpoint paths to the F07 owner explicitly. Phase 1 event-loop
regressions remain mandatory.

### 2C. Training, models, and algorithm correctness

Ownership: non-reserved `plato/trainers/**`, `plato/algorithms/**`,
`plato/models/**`, and corresponding tests. Exclude MLX and external adapter
files reserved for Phase 3; submodule source remains separately governed.

Audit sample weighting, partial adapter exchange, integer/bool state buffers,
optimizer/scheduler steps, gradient accumulation, callbacks, checkpoint state,
personalization, GAN, split learning, and differential privacy integration.
Review modern APIs where behavior benefits:

- Existing code already uses `torch.amp`; verify device/dtype/scaler correctness
  rather than apply a superficial migration.
- Consider `inference_mode` only for pure evaluation whose outputs never re-enter
  autograd; keep `no_grad` when later gradient use requires it.
- Review each `torch.load` call's data contract, `weights_only` choice and
  `map_location`. Avoid blanket unsafe loading or breaking trusted historical
  checkpoints without a migration decision.

Acceptance: failing/passing numerical reference tests for fixes, meaningful
optimizer-step assertions, save/load and continued-training equivalence where
resume is promised, and unchanged research equations/configuration meaning.
Confirmed F07 (slash model checkpoint paths), F08 (generic accumulation) and
F14 (Timm step numbering) are mandatory tasks, subject to ledger dependencies.

### 2D. Payloads, privacy, shared utilities, and cleanup

Ownership: `plato/processors/**` except Nanochat tokenizer,
`plato/serialization/**`, `plato/mpc/**`, non-reserved `plato/utils/**`,
`cleanup.py`, corresponding tests. Root assigns HE/MPC server/trainer changes
through explicit 2B/2C handoffs. Reserve `plato/utils/third_party.py` for 3B.

Audit tree/tensor serialization, compression, quantization, pruning, HE/MPC
share/sample accounting, round isolation, S3/filesystem handling, RL utilities,
and cleanup boundaries. Consider `compression.zstd` only if it reduces a real
dependency/maintenance burden while preserving a Python 3.13 fallback and wire
compatibility with existing payloads.

Acceptance: shape/dtype/tree preservation, defined tolerances for lossy
processors, malformed-input failures, correct share accounting, and cleanup
limited to intended paths. Cross-backend serialization modifications needed by
3A are coordinated here or handed off explicitly. Complete confirmed S3 tasks
F02–F05 and tree collision task F12. No protocol redesign.

Concrete security acceptance (coordinate exact file handoffs with 2A/2B/3B):

- Inventory pickle deserialization at socket report/payload handlers in
  `plato/servers/base.py`, `plato/clients/strategies/defaults.py`, and S3 reads in
  `plato/utils/s3.py`. Record that these accept only trusted peers/object writers;
  neither Safetensors inside an outer pickle nor size checks remove that trust
  requirement. Do not claim protection from arbitrary untrusted pickle input.
- Audit session-ID-to-client identity across registration, reports, chunks,
  payload completion and S3-key references. Negative tests send one client's
  claimed identity from another or unknown/stale session: reject before
  aggregation or modification of another client's state; valid sessions work.
  This validates session binding, not cryptographic authentication of peers.
- Set and exercise finite payload byte/chunk and buffered-data limits before
  deserialization/allocation where applicable. Oversized, truncated, malformed
  and incomplete messages/objects fail within a deadline, release per-session
  buffers and leave unrelated clients usable. Record actual limits and test
  rejection before unsafe parsing; do not use a successful round trip as proof.
- Run archive containment regressions on both Python 3.13 and 3.14 for
  `plato/datasources/base.py`, `purchase.py`, `texas.py`, and
  `plato/evaluators/nanochat_core.py` (3B owns the latter). Exercise TAR/ZIP
  traversal and absolute paths, TAR symlink/hardlink escapes and preexisting
  symlink destinations where supported. Assert no write outside the extraction
  root, bounded useful rejection and normal safe extraction. Do not depend on
  interpreter-specific extraction defaults for the supported-version contract.
- Phase 5 deployment documentation must describe trusted participants/storage
  and avoid implying untrusted-client safety. Authentication, an untrusted-peer
  threat model and replacement wire formats remain approval-required roadmap
  proposals; confirmed internal defects within this scope still require repair.

Phase gate: every core inventory area has a recorded outcome; confirmed
in-scope material defects have regression evidence; all retained exclusions have
specific reasons and owners. No arbitrary coverage percentage or refactor quota.

## 6. Phase 3 — MLX and external integration qualification

Prerequisites: Phase 2 gate; accepted dependency and shared payload contracts.

### 3A. MLX supported-path audit and repairs

Ownership: `plato/trainers/mlx.py`, `plato/algorithms/mlx_fedavg.py`,
`plato/models/mlx/**`, MLX branches of model/trainer/algorithm registries after
root handoff, `configs/MNIST/fedavg_lenet5_mlx.toml`, focused new MLX tests.

Root's real LeNet train/test/save/delta probe already passes. Retain this evidence
and turn the critical contracts into durable regression coverage. Only LeNet is
currently registered as an MLX model; broad backend parity is not established.

Audit:

- Backend selection and unsupported model/config combinations; no silent
  substitution of an incompatible backend.
- CPU/GPU selection, input layout conversion, train/eval modes, seeds,
  optimizer/scheduler behavior, lazy evaluation and callbacks.
- Parameter tree keys, lengths, shapes and dtypes; fix confirmed F17's unequal
  list truncation, shape broadcasting and ignored extra keys.
- Existing model checkpoints and Safetensors transport; distinguish model
  weight restoration from full optimizer/RNG resume promises.

Acceptance on the available Apple Silicon host with MLX 0.32.3:

- Real parameter-changing LeNet training and deterministic evaluation checks.
- Two independently trained MLX client models start from the same baseline,
  change parameters on their respective local data, and produce real extracted
  updates. Numerically verify their sample-weighted server aggregation and
  update application; fabricated deltas alone cannot satisfy this criterion.
- Save/load prediction equivalence and shared transport round trips.
- F17 regressions reject unequal lists, missing/extra keys and mismatched shapes
  before arithmetic/model mutation, including shapes NumPy would broadcast.
  Invalid backend selections also produce clear bounded failures.
- Documented selection/configuration commands work; absence of MLX has a useful
  error and does not break unrelated PyTorch use.
- Existing checkpoint interoperability limits are explicit. Do not introduce a
  general PyTorch-to-MLX model converter as an incidental repair.

### 3B. HuggingFace, robotics, Nanochat, and evaluators

Ownership: HuggingFace/LoRA, LeRobot/SmolVLA, and Nanochat adapter files in
`plato/{datasources,models,trainers,algorithms}/`, `plato/evaluators/**`,
`plato/processors/nanochat_tokenizer.py`, `plato/utils/third_party.py`, integration
tests, corresponding root configs, `.gitmodules` and submodule pointers by root
handoff. All relevant registry edits are serialized with 3A.

Audit real installed API compatibility as well as stub tests, adapter-only
exchange, evaluator metric interpretation, tokenizer/build dependencies, remote
asset revisions, missing-dependency failures and submodule provenance.

Acceptance:

- Tiny real HuggingFace training, PEFT adapter round trip and evaluator smoke.
- LeRobot validation in its compatible dedicated environment; do not force its
  incompatible constraints into the default environment.
- Nanochat tokenizer/build and synthetic training/evaluation exercised at the
  pinned submodule revision, with explicit prerequisites for larger workloads.
- Every optional profile has real execution evidence or a precise unresolved
  hardware/data/tooling blocker. Stub-only success is labeled accordingly.
- Submodule/license records cover Nanochat, DViT and T2T-ViT. Preserve pins unless
  an approved compatibility change is demonstrated.
- Complete confirmed F06 (all three LeRobot configs) and F09 (HuggingFace tail
  normalization), using the ledger's real config and numerical acceptance tests.
- Nanochat evaluation archive extraction passes 2D's containment regressions
  under both supported Python versions.

Phase gate: supported paths have evidence, actual capability limits are stated,
and incompatible environments remain reproducible rather than silently merged.

## 7. Phase 4 — Research examples and configuration coverage

Prerequisites: accepted core and backend phases. Audit against final interfaces.

### 4A. Active examples and root configs

Ownership: active `examples/**` Python/configuration files and remaining
`configs/**`, example-specific regression tests. Exclude `examples/outdated/**`
and third-party source updates; `examples/async/port/port_server.py` is already repaired by Phase 2/F11 and excluded from further edits until this audit identifies a new finding; manifests return to 1A/root for lock coordination.

For every entrypoint/config family in the inventory below, check import paths,
declared dependencies, registries, config includes, changed upstream APIs,
checkpoint expectations, tensor/device handling and research update semantics.
Use paper/reference equations where needed to adjudicate suspected defects.

Acceptance:

- Every entrypoint and config has a status: validated, internal-fix pending,
  external-prerequisite blocked, or explicitly archival. Internal-fix pending
  is intermediate and cannot satisfy the phase gate for a supported family.
- Each supported active family has an executed meaningful bounded behavioral
  smoke, with the actual command, environment and result retained. Parsing,
  import-only checks and an unexecuted validation command are insufficient.
  An unavailable external prerequisite leaves the family blocked and explicitly
  outside validated support; record the prerequisite and runnable command.
- Close confirmed internal bugs with reviewed fixes and executed regressions
  before accepting the affected supported family. Merely assigning a fix or
  relabeling an internal bug as an external blocker does not close it.
- Fixes preserve research semantics; performance/accuracy claims are supported
  by appropriate evidence rather than an arbitrary full benchmark campaign.
- Undeclared dependencies such as Lightly and vendored-tool imports are reviewed
  for the correct distribution/source; do not blindly install a same-named PyPI
  package, particularly NVIDIA Apex.

### 4B. Archived examples and vendored research status

Ownership: `examples/outdated/**`, archive-status records, third-party provenance
documentation after handoff to Phase 5. Vendor algorithm rewrites are out of scope.

Acceptance: all 20 archived files across `cs_maml`, `fl_maml`, FjORD and
norm-bounding have family-level status, historical dependencies/API assumptions,
known blockers and restoration estimates. Source and attribution remain intact.
Root obtains approval before restoration, removal, or semantic modernization.

Phase gate: no example family is silently omitted or described as supported
solely because its TOML parses. Every supported family passes its executed smoke
and closes its confirmed internal defects; remaining external/approval blockers
stay explicitly unresolved and cannot be counted as supported acceptance.

## 8. Phase 5 — Documentation, CI, packaging, and delivery

Prerequisites: accepted implementation/configuration interfaces from Phases 1–4.

### 5A. Documentation and support matrix

Ownership: `README.md`, `AGENTS.md`, `docs/**`, example README files and repository
templates. Coordinate any docs dependency changes with root's lock owner.

Correct the two reported MkDocs relative-link failures after capturing exact
source/target diagnostics. Reconcile installation commands, test paths,
configuration/API references, backend selection, optional environments, research
instructions, archives and third-party licensing. Resolve the 12 missing README
references in example manifests through accurate metadata or useful content.
Correct Docker documentation claiming a preconfigured environment that the
baseline Dockerfile does not provision.

Acceptance: strict docs build, valid local links/navigation, working supported
quickstarts from clean installations, clear Python/MLX/optional support matrix,
and no outdated commands claiming unsupported behavior. Review versioned docs
assets, including KaTeX, and deployment/container-toolkit instructions.
Public deployment guidance explicitly limits pickle-bearing socket/S3 paths to
trusted participants and storage writers; it must not imply untrusted-client
safety or present session binding as a complete authentication mechanism.

### 5B. CI, container tooling, and release readiness

Ownership: `.github/workflows/**`, `Dockerfile`, `dockerrun.sh`,
`dockerrun_gpu.sh`, `netlify.toml`, final acceptance evidence. Receive the minimal
workflow from 1C; root coordinates remaining shared dependency edits.

Review current authoritative action/tool/image releases and runner compatibility
before selecting versions or digests. Replace blanket all-extras/full-training
CI with explicit required CPU, optional integration, and Apple Silicon coverage;
retain real bounded startup/round checks. Validate build and publication artifacts
without actually publishing them during this task.

Acceptance:

- CI uses locked profiles and tests Python 3.13 plus the qualified default.
- Required extra jobs cannot pass solely by skipping their tests.
- CPU smoke is bounded; Apple Silicon MLX results are retained, with runner
  availability made explicit; GPU-dependent checks have a named execution path.
- Documentation, lint, typing disposition and wheel/sdist installation pass.
  Container build/startup smoke must run on an available Docker host or CI
  runner; Docker is unavailable locally, so record that check as blocked until
  executed. Known exclusions are specific and reviewable.
- Root records final dependency/vulnerability results, audit dispositions and
  reproducible commands; completes commit/push/PR workflow and task-owned cleanup.

## 9. Complete coverage inventory

At the baseline there are 886 tracked entries: 633 Python files, 57 root config
files, 471 example entries, 228 entries under `plato/`, 60 test/support entries,
49 documentation entries, and root/GitHub/submodule metadata. Counts overlap by
file type. Three tracked entries are submodule pointers, not audited upstream
source trees. There are 50 test modules and 250 statically identified test
functions; parametrization/collection explains why these are not execution counts.

### Runtime and supporting areas

| Area | Owner | Required disposition |
| --- | --- | --- |
| Root package, 15 workspace manifests, extras, lock, Python/tool versions | 1A/1C | Reproducible profile and version decisions |
| `plato/config.py`, TOML include/override handling | 2A | Semantics and isolation regressions |
| Datasources, datalib, download/extraction helpers | 2A; adapters 3B | Dataset correctness and failure behavior |
| All samplers | 2A | Partition/seed/empty-input contracts |
| Entrypoints, clients, servers, lifecycle/selection/aggregation strategies | 1B then 2B | Startup plus distributed lifecycle behavior |
| Callbacks and context/state synchronization | 2B/2C | Event and state contract checks |
| Algorithms, models, trainers, strategy families | 2C; reservations 3A/3B | Numerical/gradient/checkpoint behavior |
| DP, HE, MPC | 2C/2D with server handoff | Real required extras and protocol-preserving correctness |
| Processors, tree serialization, compression, pruning | 2D | Valid/malformed payload contracts |
| Shared utils, S3, RL policies, cleanup | 2D | Error handling, persistence and boundaries |
| MLX model/trainer/algorithm and registries | 3A | Supported-path parity and hardware regressions |
| HuggingFace, LoRA, LeRobot/SmolVLA, Nanochat | 3B | Real API and environment qualification |
| Lighteval, Nanochat CORE, evaluator registry/runner | 3B | Correct metrics and error policy |
| Tests/fixtures/fakes/global monkeypatches/skips | 1C and each component owner | Behavioral strength, isolation, truthful coverage |
| Docs, public references, deployment guidance | 5A | Working commands, links and support claims |
| Workflows, container scripts, Netlify, release artifacts | 5B | Bounded reproducible delivery |

### Active example families

Phase 4A owns each family below, with backend/component support from its earlier
owner. All families need an inventory record, not necessarily a full paper-scale
experiment.

- Basic, callbacks, customized clients/servers/processors, composable trainer.
- Async: FedAsync, FedBuff, PORT.
- Client selection: AFL, Oort, Pisces, Polaris.
- Customized training: FedDyn, FedMoS, FedProx, SCAFFOLD.
- Server aggregation: attack-adaptive, FedAdp, FedAtt, FedDF, FedNova, MOON.
- Personalized FL: APFL, Ditto, FedALA, FedAvg finetune, FedBABU, FedPer,
  FedRep, Hermes, LG-FedAvg, Per-FedAvg.
- SSL: BYOL, Calibre, FedEMA, MoCo v2, SimCLR, SimSiam, SMoG, SwAV.
- Split learning: LLM base, LoRA and attack variants.
- Secure aggregation: MaskCrypt; three-layer FL: FedSaw and Tempo.
- Unlearning: FedUnlearning and Knot, including membership-inference utilities.
- Model pruning: FedSCR and sub-FedAvg/subCS.
- Model search: AnyCostFL, FedRLNAS, FedRolex, FedTP, HeteroFL, pFedRLNAS
  DARTS/MobileNetV3/ViT, SysHeteroFL; vendored DARTS/NASViT attribution.
- Reinforcement learning: FEI; detector/poisoning defense examples.
- Gradient leakage: attacks, defenses, evaluation and pretraining utilities.
- Nanochat examples and external tokenizer requirements.

### Config, archive, documentation, and external inventory

- Root configs: CIFAR10, CIFAR100, CINIC10, CelebA, EMNIST, FEMNIST,
  FashionMNIST, HuggingFace, LeRobot, MNIST, Nanochat, Purchase100, Texas100,
  TinyImageNet and ViT. Include nested example configs and test configuration.
- Archives: `examples/outdated/cs_maml`, `fl_maml`, `model_search/fjord`,
  `norm_bounding`; 20 tracked entries total.
- Submodules: `external/nanochat` at
  `c75fe54aa7c1fa881701c246f9427bcbe4eee5a4`, `plato/models/dvit` at
  `1ccb152cea43fcbc3cd517a45c12c65734f8ace3`, and `plato/models/t2tvit` at
  `0f63dc9558f4d192de926504dbddfa1b3f5db6ca`.
- Docs: installation/quickstart, all configuration and API references, 14
  algorithm chapters, five case studies, development/deployment/CCDB/misc,
  Nanochat checklist, third-party records, MkDocs assets/navigation and Netlify.
- Root tooling: package/build metadata, `cleanup.py`, container scripts,
  `.gitignore`, `.gitmodules`, `.python-version`, license and GitHub templates.
- CI: PyTorch tests, PyPI publishing and workflow-run cleanup.

For every inventory row retain: exact paths, owner, inspected baseline,
supported environment, findings, fix/defer decision, validation command/result,
review reference and remaining limitation. An unexecuted hardware/data path is
not a passing result.

## 10. Dependency review inventory and compatibility constraints

Root's metadata snapshot has 49 records. The implementation ledger must cover
all of them and dependencies revealed by import/config inspection:

- Core: aiohttp, accelerate, boto3, datasets, evaluate, gdown, munch, numpy,
  peft, python-socketio, requests, safetensors, scipy, tenseal, timm, torch,
  torch-optimizer, torchvision and zstd.
- Optional: kazoo, gymnasium, opacus, mlx, psutil, regex, tiktoken, tokenizers,
  wandb, jinja2, PyYAML, lighteval and langdetect, plus repeated core requirements.
- Examples: scikit-learn, cvxopt, mosek, lpips, matplotlib, einops, ptflops,
  yacs, catboost and nltk.
- Build/dev/docs: hatchling, pytest, ruff, ty, uv and mkdocs-material; review
  MkDocs and resolved transitive tooling too.
- Additional imports/toolchains: Lightly, LeRobot, rustbpe/maturin/Rust, pandas,
  tqdm, torchmetrics, fvcore, progress, termcolor, scikit-image, optional
  multimedia/native dependencies and NVIDIA Apex from its actual upstream.
- Non-Python: GitHub action refs/runtime requirements, Python releases, CUDA
  images/digests, NVIDIA container tooling, KaTeX/browser assets, model/dataset
  revisions, git submodules and vendored dependencies/licenses.

Current resolved candidates include torch 2.14.1, torchvision 0.29.1,
transformers 5.18.0 and datasets 5.0.1. These require API/runtime qualification.
Lighteval 0.13.0 is still its current release; retain its pin unless evidence
supports changing it. LeRobot 0.6.1 declares torch `<2.12` and numpy `<2.3`, so
its compatibility profile must remain distinct from the current default stack.

Planning-time upstream release observations, not automatic upgrade instructions:
checkout v7.0.1, setup-python v7.0.0, setup-uv v10.2.0,
delete-workflow-runs v2.1.0, and libnvidia-container v1.20.1. Recheck metadata and
runner/platform requirements when implementing delivery changes.

## 11. Proposed MLX expansion roadmap — approval required

The refresh includes the supported-path repairs and regression coverage in 3A.
Additional work is prioritized as follows, but is not authorized by this plan:

1. Broader optimizer/scheduler and full resumable-state parity; a second useful
   vision model; sustained Apple Silicon CI. Acceptance would require numerical
   step/resume comparisons, a real federated smoke and maintained support docs.
2. Explicit PyTorch/MLX checkpoint conversion for matching architectures.
   Acceptance would require documented parameter/layout mapping and prediction
   equivalence in both directions. Shared Safetensors encoding alone does not
   provide model architecture or optimizer-state interoperability.
3. More federated strategies and MLX-native language-model/LoRA workloads,
   selected by research demand and measured benefit. Define individual model,
   dataset, algorithm, hardware and performance targets before implementation.

Do not silently broaden LeNet/FedAvg repair into complete backend parity,
cross-backend federated training, or a new transport/checkpoint protocol.

## 12. Risks, blockers, and stopping conditions

- **Immediate gate:** independent plan review is pending. Root requests and
  resolves it before implementation.
- **Python 3.14 runtime blocker:** synchronous loop bootstrap fails despite a
  passing suite. Task 1B must close it before qualification/default promotion.
- **Reproducibility gap:** generated local lock is not tracked at baseline.
- **Remote workflow available:** GitHub app push/admin/PR capabilities and SSH
  origin reachability resolve the prior publication blocker. `gh` authentication
  is not required by the selected workflow.
- **Container validation blocker:** Docker is unavailable locally. Use an
  available runner/host before claiming container acceptance; static checks
  alone do not close this requirement.
- **Compatibility risks:** large upstream API changes, NumPy/dtype behavior,
  checkpoint loading defaults, process start methods, native wheels, incompatible
  robotics constraints and research/vendor APIs.
- **Validation limits:** Apple Silicon is available and MLX probes pass; GPU,
  licensed MOSEK, ZooKeeper/S3, gated datasets/models, and external tokenizer
  toolchains may need dedicated execution resources. Record actual unavailable
  prerequisites rather than infer failures or mark them tested.
- **Submodule prerequisite resolved:** all three submodules are initialized at
  their pinned SHAs; qualification of their runtime behavior is still required.
- **Baseline coverage:** 43% and stub-heavy integration areas mean a green suite
  is incomplete evidence. Add tests around demonstrated behavior gaps, not a
  cosmetic coverage target.
- **Scope conflicts:** ask the user before research semantic changes, transport
  or authentication redesign, larger MLX features, archive restoration/removal,
  or any proposal to
  drop the retained Python 3.13 minimum.
- **Bounded completion:** all inventory entries receive a disposition; supported
  profiles pass their acceptance checks; accepted material defects are fixed;
  unresolved external or scope blockers are explicit. Do not claim unsupported
  profiles are complete, and do not continue speculative refactoring to chase
  unrelated improvements.
