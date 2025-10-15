• Client API Plan

  - plato/clients/base.py:78 mixes connection orchestration, payload IO, processor hooks, training dispatch, and async update handling; subclasses like simple.Client (plato/clients/simple.py:22)
    override several unrelated methods, creating tight coupling.
  - Pipeline stages—data prep (plato/clients/simple.py:99), trainer/algorithm wiring (plato/clients/simple.py:54), payload transformation (plato/clients/base.py:213), and report generation (plato/
    clients/simple.py:193)—are intertwined with socket events (plato/clients/base.py:181), making reuse hard.
  - Specializations (edge, split, MistNet) override different slices of the lifecycle, indicating natural seams for strategies: data allocation, payload interpretation, training execution, report
    customization, and async retrieval (plato/clients/simple.py:206).
  - The new server/trainer APIs rely on lightweight contexts plus swappable strategy interfaces, which we can mirror to isolate client responsibilities and reduce inheritance depth.

  Proposed Design

  - ClientContext: shared state (client id, round, datasource, trainer, processors, timers, comm options) passed to strategies; mirrors TrainingContext but adds networking + async metadata.
  - LifecycleStrategy: coordinates configure/load/allocate steps, allowing custom datasource or sampler lifecycles; default implementation wraps current configure/_load_data/_allocate_data.
  - PayloadStrategy: handles inbound processing, payload deserialization, and outbound packaging (current _handle_payload, _send, chunking logic); supports variants like split learning features or
    gradient payloads.
  - TrainingStrategy: encapsulates _load_payload, _train, and trainer invocation; default uses registry-based trainer/algorithm, while MistNet/Split provide custom subclasses.
  - ReportingStrategy: builds and post-processes reports (current customize_report, async _obtain_model_at_time), including sleep simulation and metadata management.
  - CommunicationStrategy: abstracts simulation vs socket/S3 transfer, managing chunk size, retry, and metrics; keeps socket event wiring in one place and exposes hooks for future transports.
  - All strategies derive from a ClientStrategy base with setup/teardown, enabling composition in a new ComposableClient similar to ComposableTrainer.

  Migration Steps

  - Introduce plato/clients/strategies with base interfaces, ClientContext, and default implementations that reproduce today’s behaviour exactly.
  - Create ComposableClient that wires strategies, registers socket events, and orchestrates the round lifecycle; keep legacy base.Client temporarily delegating to the composable core for backward
    compatibility.
  - Convert simple.Client to a thin configuration of default strategies; port specialized clients by swapping/stacking strategies instead of overriding large methods.
  - Update plato/clients/registry.py to instantiate ComposableClient configurations, provide compatibility shims for external subclasses, and extend docs/examples to the new API.
  - Add regression tests covering payload round-trips, async model requests, and strategy swapping; ensure existing integration tests run against both legacy and composable paths during the
    transition.

    - Migrate the split-learning pipeline from the legacy stack to purpose-built strategies.
    - Audit remaining specialised clients for any direct attribute tweaks and replace them with context-driven strategy adaptations.

  Next Steps

  1. Align on strategy boundaries/naming and document the default behaviours for review.
  2. Prototype the ClientContext + default strategies using the current simple client to validate parity.
  3. Plan incremental migration/testing milestones (e.g., migrate simple + edge first, then split/mistnet).

• Findings

  - Core runtime now follows the plan’s composition design: ClientContext plus strategy interfaces live in plato/clients/strategies/base.py:17-
    223, base.Client wires them into a ComposableClient bridge plato/clients/base.py:30-144, and the orchestrator drives lifecycle/communication
    events plato/clients/composable.py:217-334.
  - Default stacks back simple.Client and the in-tree specialisations—edge, MistNet, split—through dedicated strategy classes, with the registry
    falling back to legacy adapters when a subclass does not self-configure (plato/clients/simple.py:15-111, plato/clients/strategies/edge.py:21-
    109, plato/clients/strategies/mistnet.py:20-47, plato/clients/strategies/split_learning.py:1-142, plato/clients/registry.py:1-125).
  - However, the migration step to “audit remaining specialised clients” is still open: the default lifecycle strategy’s process_server_response
    is a no-op (plato/clients/strategies/defaults.py:43-51), yet many example/custom clients still rely on the old override (e.g. examples/
    model_search/heterofl/heterofl_client.py:9-17, examples/unlearning/knot/knot_client.py:14-36), so they now skip required behaviour unless
    refactored into strategies or reattached to the legacy shim (docs/docs/client-redesign/plan.md:32-33).
  - The plan also called for regression coverage of payload round-trips, async requests, and strategy swaps (docs/docs/client-redesign/
    plan.md:24-30), but the test suite still only exercises the legacy-style direct hooks via tests/client_tests.py:13-31, leaving the composable
    workflow untested.
