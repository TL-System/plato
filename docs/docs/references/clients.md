# Clients

## Strategy-Based Client Architecture

Plato's client runtime now centres on a composable pipeline rather than deeply
nested subclasses. Every `plato.clients.base.Client` instance owns a
`ComposableClient` (`plato/clients/composable.py`) that orchestrates five
pluggable strategies:

- `LifecycleStrategy` prepares the datasource, trainer, and samplers.
- `PayloadStrategy` rebuilds inbound payloads and prepares outbound data.
- `TrainingStrategy` loads weights and runs the local optimisation loop.
- `ReportingStrategy` finalises metadata and serves asynchronous requests.
- `CommunicationStrategy` serialises reports/payloads for transport.

Shared state flows between these strategies through `ClientContext`
(`plato/clients/strategies/base.py`). The context mirrors historically mutable
attributes—client id, datasource, processors, timers, and callbacks—so the
strategies can collaborate without touching private attributes on the client.

The default stack (`Default*Strategy` in
`plato/clients/strategies/defaults.py`) reproduces the legacy behaviour that
powered `simple.Client`. Specialised presets build on top of the same base,
for example `EdgeLifecycleStrategy` or `MistNetTrainingStrategy`.

## Composing Clients

The reference implementation in `plato/clients/simple.py` illustrates how to
assemble a strategy-based client: configure custom factories on the context,
then call `_configure_composable(...)` with the desired strategy instances.
Only the strategies you swap need new code—inherit the defaults elsewhere.

```py
from plato.clients import base
from plato.clients.strategies import (
    DefaultCommunicationStrategy,
    DefaultLifecycleStrategy,
    DefaultReportingStrategy,
    DefaultTrainingStrategy,
)
from plato.clients.strategies.defaults import DefaultPayloadStrategy


class AugmentedPayloadStrategy(DefaultPayloadStrategy):
    def outbound_ready(self, context, report, outbound_payload):
        super().outbound_ready(context, report, outbound_payload)
        report.extra_metrics = context.metadata.get("custom_metrics", {})


class VisionClient(base.Client):
    def __init__(self, *, callbacks=None):
        super().__init__(callbacks=callbacks)
        self._configure_composable(
            lifecycle_strategy=DefaultLifecycleStrategy(),
            payload_strategy=AugmentedPayloadStrategy(),
            training_strategy=DefaultTrainingStrategy(),
            reporting_strategy=DefaultReportingStrategy(),
            communication_strategy=DefaultCommunicationStrategy(),
        )
```

Within a strategy you receive a `ClientContext` rather than the client
instance. This makes it straightforward to compose behaviour:

- Inspect or mutate `context.sampler`, `context.datasource`, or
  `context.trainset` during `LifecycleStrategy.allocate_data`.
- Share intermediate values via `context.state` and expose round metadata
  through `context.metadata`.
- Call `context.callback_handler.call_event(...)` to reuse the existing
  callback pipeline whenever you add new strategy events.

Remember to synchronise any long-lived fields back to the owner if you change
them in place (see `ComposableClient._sync_owner_from_context` for reference).

## Strategy Extension Points

- **`LifecycleStrategy`** (`plato/clients/strategies/base.py`) governs
  configuration. Override:
  - `process_server_response(context, server_response)` to populate round
    metadata or react to scheduler hints.
  - `load_data(context)` to build datasources or skip them for proxy clients.
  - `configure(context)` to instantiate trainers/algorithms/processors.
  - `allocate_data(context)` to wire samplers and train/test partitions.
  The defaults fetch registry components and honour config flags such as
  `clients.do_test`.

- **`PayloadStrategy`** coordinates payload reconstruction. Reuse the default
  for pickled model weights, or override:
  - `accumulate_chunk` / `commit_chunk_group` for multi-part transfers.
  - `finalise_inbound_payload` when downloading from external storage (S3,
    split learning, etc.).
  - `handle_server_payload` to apply custom preprocessing before training.

- **`TrainingStrategy`** encapsulates weight loading and the local optimisation
  loop. Implement `load_payload` and `train`; the default delegates to the
  configured algorithm and trainer while respecting optional evaluation
  (`clients.do_test`, `clients.test_interval`).

- **`ReportingStrategy`** finalises metadata. Override `build_report` to enrich
  the report before it leaves the client, or customise
  `obtain_model_at_time` to serve asynchronous updates (used by draw-and-discard
  or MistNet workloads).

- **`CommunicationStrategy`** handles transport. The default emits Socket.IO
  events and optionally uploads to S3, but you can substitute a strategy for
  alternative channels (file system, RPC, simulated environments) by replacing
  `send_report` and `send_payload`.

Each strategy exposes optional `setup`/`teardown` hooks; use them to allocate
resources when the client boots or release them once the round finishes.

## Backwards Compatibility Hooks

Existing subclasses that overrode the legacy methods—`configure`,
`process_server_response`, `_load_data`, `_allocate_data`,
`inbound_processed`, and friends—still function. `base.Client` now attaches
`Legacy*Strategy` adapters (`plato/clients/strategies/legacy.py`) that forward
strategy calls into those overrides. This safety net keeps historical clients
operational, but new development should migrate into dedicated strategies so
behaviour is explicit and reusable.

If you gradually port an existing client, you can mix approaches: keep the
legacy adapters for the parts you have not touched yet, and replace individual
strategies once they have been rewritten.

## Client Callbacks

Callbacks remain the preferred way to inject cross-cutting concerns such as
logging, tracing, or metrics aggregation. Subclass
`plato.callbacks.client.ClientCallback`, implement the relevant
`on_inbound_received`, `on_inbound_processed`, or `on_outbound_ready` hooks,
and pass the callback class to the client constructor (or call
`client.add_callbacks`).

The callback handler is stored on `ClientContext.callback_handler`, so strategy
implementations can continue to fire the same events that legacy clients used.
When designing new strategies, invoke the handler to keep observability
features working for downstream experiments.
