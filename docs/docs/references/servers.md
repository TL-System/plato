# Servers

## Strategy-Based Server Architecture

Plato servers now support strategy-based composition for the two most common customization points:
client selection and update aggregation. Instead of subclassing the server and overriding hooks, you
can pass strategy objects that implement lightweight interfaces.

### Overview

- `AggregationStrategy`: orchestrates how client model updates are merged into the global model.
- `ClientSelectionStrategy`: decides which clients participate in each round.
- `ServerStrategy`: shared base class that exposes lifecycle hooks for setup/teardown.
- `ServerContext`: shared state passed to every strategy so they can coordinate without tight
  coupling to the concrete server implementation.

Strategy instances can be combined at runtime, making it easy to mix built-in functionality with your
own components.

### Quick Start

```py
from plato.servers import fedavg
from plato.servers.strategies.aggregation import FedNovaAggregationStrategy
from plato.servers.strategies.client_selection import OortSelectionStrategy

server = fedavg.Server(
    aggregation_strategy=FedNovaAggregationStrategy(),
    client_selection_strategy=OortSelectionStrategy(exploration_factor=0.3),
)

server.run()
```

If you only need to customize one side, pass the other strategy as `None` and the server falls back
to its default implementation.

### Built-in Strategies

| Strategy type | Class | Highlights |
| --- | --- | --- |
| Aggregation | `FedAvgAggregationStrategy` | Sample-weighted FedAvg implementation. |
| Aggregation | `FedAsyncAggregationStrategy` | Staleness-aware mixing for asynchronous training. |
| Aggregation | `FedBuffAggregationStrategy` | Simple asynchronous aggregation strategy without using weights. |
| Aggregation | `FedNovaAggregationStrategy` | Normalized FedNova variant for heterogeneous local epochs. |
| Aggregation | `HermesAggregationStrategy` | Mask-aware aggregation used by the Hermes personalization algorithm. |
| Aggregation | `FedAvgGanAggregationStrategy` | Generator/discriminator-aware averaging for GAN training. |
| Aggregation | `FedAvgHEAggregationStrategy` | Hybrid encrypted/plain averaging for CKKS-based workflows. |
| Client selection | `RandomSelectionStrategy` | Uniform random selection (default). |
| Client selection | `SplitLearningSequentialSelectionStrategy` | Sequentially serves one client at a time for split learning. |
| Client selection | `PersonalizedRatioSelectionStrategy` | Limits participation by ratio before a personalization phase. |

### Implementing Custom Strategies

```py
from typing import Dict, List
from types import SimpleNamespace

from plato.servers.strategies.base import (
    AggregationStrategy,
    ClientSelectionStrategy,
    ServerContext,
)


class ClippedAggregationStrategy(AggregationStrategy):
    """Clip client deltas before averaging to improve robustness."""

    def __init__(self, max_norm: float = 5.0):
        self.max_norm = max_norm

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        total_samples = sum(update.report.num_samples for update in updates)
        averaged = {
            name: context.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        for i, delta in enumerate(deltas_received):
            weight = updates[i].report.num_samples / total_samples
            for name, value in delta.items():
                clipped = value.clamp(-self.max_norm, self.max_norm)
                averaged[name] += clipped * weight

        return averaged


class StragglerAwareSelection(ClientSelectionStrategy):
    """Avoid repeatedly selecting clients that recently participated."""

    def select_clients(
        self,
        clients_pool: List[int],
        clients_count: int,
        context: ServerContext,
    ) -> List[int]:
        history = context.state.setdefault(
            "recent_clients", []
        )  # maintain simple FIFO history
        eligible = [cid for cid in clients_pool if cid not in history]

        if len(eligible) < clients_count:
            eligible = clients_pool  # fallback if pool is too small

        selected = eligible[:clients_count]
        history.extend(selected)
        history[:] = history[-2 * clients_count :]  # keep last few entries
        return selected
```

Key tips:

- Strategies receive a `ServerContext` instance on every call; use it to read or share runtime state.
- Aggregation strategies may override `aggregate_weights()` when working with weight dictionaries
  instead of deltas.
- Client selection strategies can optionally implement `on_clients_selected()` and
  `on_reports_received()` hooks when additional bookkeeping is required.

### Strategy Interfaces

`plato/servers/strategies/base.py` defines the shared contracts. The most important attributes on
`ServerContext` are:

- `server`, `trainer`, and `algorithm` references for interacting with the broader system.
- `current_round`, `total_clients`, and `clients_per_round` counters.
- `updates`: list of `SimpleNamespace` instances containing the latest batch of client reports.
- `state`: dictionary for persisting cross-call state without mutating the server directly.

Refer to the source docstrings for the complete interface.

### Migrating from Hook Overrides

The hook-based approach, as documented in the next section, continues to work for advanced scenarios. We recommend the strategy pattern for new projects because it keeps responsibilities modular and testable. When migrating:

1. Identify the overridden hook (for example, `choose_clients`) and map it to the corresponding strategy (`ClientSelectionStrategy.select_clients`).
2. Move helper attributes into the strategy's internal state or the shared `context.state`.
3. Register the strategy in your server factory or experiment script.

## Customizing Servers using Subclassing

The common practice is to customize the server using subclassing for important features that change the state of the server. To customize the server using inheritance, subclass the `fedavg.Server` (or `fedavg_cs.Server` for cross-silo federated learning) class in `plato.servers`, and override the following methods:

!!! example "configure()"
    **`def configure(self) -> None`**

    Override this method to implement additional tasks for initializing and configuring the server. Make sure that `super().configure()` is called first.

    **Example:**

    ```py
    def configure(self) -> None:
        """Configure the model information like weight shapes and parameter numbers."""
        super().configure()

        self.total_rounds = Config().trainer.rounds
    ```

!!! example "init_trainer()"
    **`def init_trainer(self) -> None`**

    Override this method to implement additional tasks for initializing and configuring the trainer. Make sure that `super().init_trainer()` is called first.

    **Example** (from `examples/knot/knot_server.py`):

    ```py
    def init_trainer(self) -> None:
        """Load the trainer and initialize the dictionary that maps cluster IDs to client IDs."""
        super().init_trainer()

        self.algorithm.init_clusters(self.clusters)
    ```

!!! example "choose_clients()"
    **`def choose_clients(self, clients_pool, clients_count)`**

    Override this method to implement a customized client selection algorithm, choosing a subset of clients from the client pool.

    `clients_pool` a list of available clients for selection.

    `clients_count` the number of clients that need to be selected in this round.

    When overriding this method, delegate to `_select_clients_with_strategy()` if you only need to filter the candidate pool. This keeps the strategy stack (and reproducible random state) in sync with the rest of the server.

    ```py
    def choose_clients(self, clients_pool, clients_count):
        filtered = [cid for cid in clients_pool if cid not in self.blacklist]
        return self._select_clients_with_strategy(filtered, clients_count)
    ```

    **Returns:** a list of selected client IDs.

!!! example "weights_received()"
    **`def weights_received(self, weights_received)`**

    Override this method to complete additional tasks after the updated weights have been received.

    `weights_received` the updated weights that have been received from the clients.

    **Example:**

    ```py
    def weights_received(self, weights_received):
        """
        Event called after the updated weights have been received.
        """
        self.control_variates_received = [weight[1] for weight in weights_received]
        return [weight[0] for weight in weights_received]
    ```

!!! example "aggregate_deltas()"
    **`async def aggregate_deltas(self, updates, deltas_received)`**

    In most cases, it is more convenient to aggregate the model deltas from the clients, because this can be performed in a framework-agnostic fashion. Override this method to aggregate the deltas received. This method is needed if `aggregate_weights()` (below) is not defined.

    `updates` the client updates received at the server.

    `deltas_received` the weight deltas received from the clients.

!!! example "aggregate_weights()"
    **`async def aggregate_weights(self, updates, baseline_weights, weights_received)`**

    Sometimes it is more convenient to aggregate the received model weights directly to the global model. In this case, override this method to aggregate the weights received directly to baseline weights. This method is optional, and the server will call this method rather than `aggregate_deltas` when it is defined. Refer to `examples/fedasync/fedasync_server.py` for an example.

    `updates` the client updates received at the server.

    `baseline_weights` the current weights in the global model.

    `weights_received` the weights received from the clients.

!!! example "weights_aggregated()"
    **`def weights_aggregated(self, updates)`**

    Override this method to complete additional tasks after aggregating weights.

    `updates` the client updates received at the server.

!!! example "customize_server_response()"
    **`def customize_server_response(self, server_response: dict, client_id) -> dict`**

    Override this method to return a customize server response with any additional information.

    `server_response` key-value pairs (from a string to an instance) for the server response before customization.

    `client_id` the client ID.

    **Example:**

    ```py
    def customize_server_response(self, server_response: dict, client_id) -> dict:
        """
        Customizes the server response with any additional information.
        """
        server_response["pruning_amount"] = self.pruning_amount_list
        return server_response
    ```

!!! example "customize_server_payload()"
    **`def customize_server_payload(self, payload)`**

    Override this method to customize the server payload before sending it to the clients.

    **Returns:** Customized server payload to be sent to the clients.

!!! example "clients_selected()"
    **`def clients_selected(self, selected_clients) -> None`**

    Override this method to complete additional tasks after clients have been selected in each round.

    `selected_clients` a list of client IDs that have just been selected by the server.

!!! example "clients_processed()"
    **`def clients_processed(self) -> None`**

    Override this method to complete additional tasks after all client reports have been processed.

!!! example "get_logged_items()"
    **`def get_logged_items(self) -> dict`**

    Override this method to return items to be logged by the `LogProgressCallback` class in a .csv file.

    **Returns:** a dictionary of items to be logged.

    **Example:** (from `examples/knot/knot_server`)

    ```py
    def get_logged_items(self):
        """Get items to be logged by the LogProgressCallback class in a .csv file."""
        logged_items = super().get_logged_items()

        clusters_accuracy = [
            self.clustered_test_accuracy[cluster_id]
            for cluster_id in range(self.num_clusters)
        ]

        clusters_accuracy = "; ".join([str(acc) for acc in clusters_accuracy])

        logged_items["clusters_accuracy"] = clusters_accuracy

        return logged_items
    ```

!!! example "should_request_update()"
    **`def should_request_update(self, client_id, start_time, finish_time, client_staleness, report):`**

    Override this method to save additional information when the server saves checkpoints at the end of each around.

    `client_id` The client ID for the client to be considered.

    `start_time` The wall-clock time when the client started training.

    `finish_time` The wall-clock time when the client finished training.

    `client_staleness` The number of rounds that elapsed since this client started training.

    `report` The report sent by the client.

    **Returns:** `True` if the server should explicitly request an update from the client `client_id`; `False` otherwise.

    **Example:** (from `servers/base.py`)
    ```py
        def should_request_update(
            self, client_id, start_time, finish_time, client_staleness, report
        ):
            """Determines if an explicit request for model update should be sent to the client."""
            return client_staleness > self.staleness_bound and finish_time > self.wall_time
    ```

!!! example "save_to_checkpoint()"
    **`def save_to_checkpoint(self) -> None`**

    Override this method to save additional information when the server saves checkpoints at the end of each around.

!!! example "training_will_start()"
    **`def training_will_start(self) -> None`**

    Override this method to complete additional tasks before selecting clients for the first round of training.

!!! example "periodic_task()"
    **`periodic_task(self) -> None`**

    Override this async method to perform periodic tasks in asynchronous mode, where this method will be called periodically.

!!! example "wrap_up()"
    **`async def wrap_up(self) -> None`**

    Override this method to complete additional tasks at the end of each round.

!!! example "server_will_close()"
    **`def server_will_close(self) -> None:`**

    Override this method to complete additional tasks before closing the server.

## Customizing Servers using Callbacks

For infrastructure changes, such as logging and recording metrics, we tend to customize the global training process using callbacks instead. The advantage of using callbacks is that one can pass a list of multiple callbacks to the server when it is initialized, and they will be called in their order in the provided list. This helps when it is necessary to group features into different callback classes.

Within the implementation of these callback methods, one can access additional information about the global training by using the `server` instance.

To use callbacks, subclass the `ServerCallback` class in `plato.callbacks.server`, and override the following methods, then pass it to the server when it is initialized, or call `server.add_callbacks` after initialization. Examples can be found in `examples/callbacks`.

!!! example "on_weights_received()"
    **`def on_weights_received(self, server, weights_received)`**

    Override this method to complete additional tasks after the updated weights have been received.

    `weights_received` the updated weights that have been received from the clients.

!!! example "on_weights_aggregated()"
    **`def on_weights_aggregated(self, server, updates)`**

    Override this method to complete additional tasks after aggregating weights.

    `updates` the client updates received at the server.

    **Example:**

    ```py
    def on_weights_aggregated(self, server, updates):
        logging.info("[Server #%s] Finished aggregating weights.", os.getpid())
    ```

!!! example "on_clients_selected()"
    **`def on_clients_selected(self, server, selected_clients)`**

    Override this method to complete additional tasks after clients have been selected in each round.

    `selected_clients` a list of client IDs that have just been selected by the server.

!!! example "on_clients_processed()"
    **`def on_clients_processed(self, server)`**

    Override this method to complete additional tasks after all client reports have been processed.

!!! example "on_training_will_start()"
    **`def on_training_will_start(self, server)`**

    Override this method to complete additional tasks before selecting clients for the first round of training.

!!! example "on_server_will_close()"
    **`def on_server_will_close(self, server)`**

    Override this method to complete additional tasks before closing the server.

    **Example:**

    ```py
    def on_server_will_close(self, server):
        logging.info("[Server #%s] Closing the server.", os.getpid())
    ```
