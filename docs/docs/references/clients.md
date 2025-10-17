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
for example `EdgeLifecycleStrategy`.

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
  `obtain_model_at_time` to serve asynchronous updates.

- **`CommunicationStrategy`** handles transport. The default emits Socket.IO
  events and optionally uploads to S3, but you can substitute a strategy for
  alternative channels (file system, RPC, simulated environments) by replacing
  `send_report` and `send_payload`.

Each strategy exposes optional `setup`/`teardown` hooks; use them to allocate
resources when the client boots or release them once the round finishes.

## Strategy Recipes in Examples

Most example workloads now customise clients by swapping strategies instead of
overriding legacy hooks. The following patterns provide practical templates:

- **Reporting augmentations.** `OortReportingStrategy`,
  `PiscesReportingStrategy`, and `AFLReportingStrategy` (under
  `examples/client_selection/`) compute statistical utility or valuations
  inside `build_report`, pulling metrics from trainer run histories and
  tolerating missing data.
- **Lifecycle configuration.** `ScaffoldLifecycleStrategy`
  (`examples/customized_client_training/scaffold/scaffold_client.py`) restores
  persisted control variates, while `FedNovaLifecycleStrategy`
  (`examples/server_aggregation/fednova/fednova_client.py`) performs per-client
  RNG seeding during `configure`.
- **Training specialisation.** Strategies such as `DLGTrainingStrategy`
  (`examples/gradient_leakage_attacks/dlg_client.py`), `SubFedAvgTrainingStrategy`
  (`examples/model_pruning/sub_fedavg/subfedavg_client.py`), and
  `FedSawTrainingStrategy` / `FedSawEdgeTrainingStrategy`
  (`examples/three_layer_fl/fedsaw/`) show how to augment payloads, add logging,
  or prune updates before transmission. `FlMamlTrainingStrategy`
  (`examples/outdated/fl_maml/fl_maml_client.py`) illustrates personalised
  evaluation flows.
- **Metadata propagation.** NAS and pruning case studies attach algorithm state
  to reports via strategy overrides (`FedRLNASReportingStrategy`,
  `PerFedRLNASReportingStrategy`, `FedSCRReportingStrategy`, etc.), eliminating
  the need to mutate the client directly.
- **Edge coordination.** Cross-silo scenarios extend
  `EdgeTrainingStrategy`: see `CsMamlEdgeTrainingStrategy` and
  `FedSawEdgeTrainingStrategy` for examples that add personalisation tests or
  post-aggregation pruning.

Copy one of these strategies, tailor the hook you need, then wire it into
`_configure_composable(...)` on your client subclass.

## Migration Notes

The legacy adapter layer (`plato/clients/strategies/legacy.py`) has been
removed. Client subclasses must configure strategies explicitly via
`_configure_composable(...)`; overriding legacy hooks such as `_train` or
`_load_payload` no longer affects the runtime.

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


### Customizing Clients using Callbacks

For infrastructure changes, such as logging and recording metrics, we tend to customize the client using callbacks instead. The advantage of using callbacks is that one can pass a list of multiple callbacks to the client when it is initialized, and they will be called in their order in the provided list. This helps when it is necessary to group features into different callback classes.

Within the implementation of these callback methods, one can access additional information about the local training by using the `client` instance.

To use callbacks, subclass the `ClientCallback` class in `plato.callbacks.client`, and override the following methods, then pass it to the client when it is initialized, or call `client.add_callbacks` after initialization. Examples can be found in `examples/callbacks`.


!!! example "on_inbound_received()"
    **`def on_inbound_received(self, client, inbound_processor)`**

    Override this method to complete additional tasks before the inbound processors start to process the data received from the server.

    `inbound_processor` the pipeline of inbound processors. The list of inbound processor instances can be accessed through its attribute 'processors'.


!!! example "on_inbound_processed()"
    **`def on_inbound_processed(self, client, data)`**

    Override this method to complete additional tasks when the inbound data has been processed by inbound processors.

    `data` the inbound data after being processed by inbound processors, e.g., model weights before loaded to the trainer.

    **Example:**

    ```py
    def on_inbound_processed(self, client, data):
        # print the layer names of the model weights before further operations
        for name, weights in data:
            print(name)
    ```

!!! example "on_outbound_ready()"
    **`def on_outbound_ready(self, client, outbound_processor)`**

    Override this method to complete additional tasks before the outbound processors start to process the data to be sent to the server.

    `outbound_processor` the pipeline of outbound processors. The list of inbound processor instances can be accessed through its attribute 'processors'.
