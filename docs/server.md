# Servers

## Customizing servers using inheritance

The common practice is to customize the server using inheritance for important features that change the state of the server. To customize the server using inheritance, subclass the `fedavg.Server` (or `fedavg_cs.Server` for cross-silo federated learning) class in `plato.servers`, and override the following methods:

````{admonition} **configure()**
**`def configure(self) -> None`**

Override this method to implement additional tasks for initializing and configuring the server. Make sure that `super().configure()` is called first.

**Example:**

```py
def configure(self) -> None:
    """Configure the model information like weight shapes and parameter numbers."""
    super().configure()

    self.total_rounds = Config().trainer.rounds
```
````

````{admonition} **init_trainer()**
**`def init_trainer(self) -> None`**

Override this method to implement additional tasks for initializing and configuring the trainer. Make sure that `super().init_trainer()` is called first.

**Example** (from `examples/knot/knot_server.py`):

```py
def init_trainer(self) -> None:
    """Load the trainer and initialize the dictionary that maps cluster IDs to client IDs."""
    super().init_trainer()

    self.algorithm.init_clusters(self.clusters)
```
````

```{admonition} **choose_clients()**
**`def choose_clients(self, clients_pool, clients_count)`**

Override this method to implement a customized client selection algorithm, choosing a subset of clients from the client pool.

`clients_pool` a list of available clients for selection.

`clients_count` the number of clients that need to be selected in this round.

**Returns:** a list of selected client IDs.
```

````{admonition} **weights_received()**
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
````

```{admonition} **aggregate_deltas()**
**`async def aggregate_deltas(self, updates, deltas_received)`**

In most cases, it is more convenient to aggregate the model deltas from the clients, because this can be performed in a framework-agnostic fashion. Override this method to aggregate the deltas received. This method is needed if `aggregate_weights()` (below) is not defined.

`updates` the client updates received at the server.

`deltas_received` the weight deltas received from the clients.
```

```{admonition} **aggregate_weights()**
**`async def aggregate_weights(self, updates, baseline_weights, weights_received)`**

Sometimes it is more convenient to aggregate the received model weights directly to the global model. In this case, override this method to aggregate the weights received directly to baseline weights. This method is optional, and the server will call this method rather than `aggregate_deltas` when it is defined. Refer to `examples/fedasync/fedasync_server.py` for an example.

`updates` the client updates received at the server.

`baseline_weights` the current weights in the global model.

`weights_received` the weights received from the clients.
```

````{admonition} **weights_aggregated()**
**`def weights_aggregated(self, updates)`**

Override this method to complete additional tasks after aggregating weights.

`updates` the client updates received at the server.
````

````{admonition} **customize_server_response()**
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
````

```{admonition} **customize_server_payload()**
**`def customize_server_payload(self, payload)`**

Override this method to customize the server payload before sending it to the clients.

**Returns:** Customized server payload to be sent to the clients.
```

```{admonition} **clients_selected()**
**`def clients_selected(self, selected_clients) -> None`**

Override this method to complete additional tasks after clients have been selected in each round.

`selected_clients` a list of client IDs that have just been selected by the server.
```

```{admonition} **clients_processed()**
**`def clients_processed(self) -> None`**

Override this method to complete additional tasks after all client reports have been processed.
```

````{admonition} **get_logged_items()**
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
````

````{admonition} **should_request_update()**
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
````

```{admonition} **save_to_checkpoint()**
**`def save_to_checkpoint(self) -> None`**

Override this method to save additional information when the server saves checkpoints at the end of each around.
```

```{admonition} **training_will_start()**
**`def training_will_start(self) -> None`**

Override this method to complete additional tasks before selecting clients for the first round of training.
```

```{admonition} **periodic_task()**
**`periodic_task(self) -> None`**

Override this async method to perform periodic tasks in asynchronous mode, where this method will be called periodically.
```

```{admonition} **wrap_up()**
**`async def wrap_up(self) -> None`**

Override this method to complete additional tasks at the end of each round.
```

```{admonition} **server_will_close()**
**`def server_will_close(self) -> None:`**

Override this method to complete additional tasks before closing the server.
```

## Customizing servers using callbacks

For infrastructure changes, such as logging and recording metrics, we tend to customize the global training process using callbacks instead. The advantage of using callbacks is that one can pass a list of multiple callbacks to the server when it is initialized, and they will be called in their order in the provided list. This helps when it is necessary to group features into different callback classes.

Within the implementation of these callback methods, one can access additional information about the global training by using the `server` instance. 

To use callbacks, subclass the `ServerCallback` class in `plato.callbacks.server`, and override the following methods, then pass it to the server when it is initialized, or call `server.add_callbacks` after initialization. Examples can be found in `examples/callbacks`.

````{admonition} **on_weights_received()**
**`def on_weights_received(self, server, weights_received)`**

Override this method to complete additional tasks after the updated weights have been received.

`weights_received` the updated weights that have been received from the clients.
````

````{admonition} **on_weights_aggregated()**
**`def on_weights_aggregated(self, server, updates)`**

Override this method to complete additional tasks after aggregating weights.

`updates` the client updates received at the server.

**Example:**

```py
def on_weights_aggregated(self, server, updates):
    logging.info("[Server #%s] Finished aggregating weights.", os.getpid())
```
````

```{admonition} **on_clients_selected()**
**`def on_clients_selected(self, server, selected_clients)`**

Override this method to complete additional tasks after clients have been selected in each round.

`selected_clients` a list of client IDs that have just been selected by the server.
```

```{admonition} **on_clients_processed()**
**`def on_clients_processed(self, server)`**

Override this method to complete additional tasks after all client reports have been processed.
```

```{admonition} **on_training_will_start()**
**`def on_training_will_start(self, server)`**

Override this method to complete additional tasks before selecting clients for the first round of training.
```

````{admonition} **on_server_will_close()**
**`def on_server_will_close(self, server)`**

Override this method to complete additional tasks before closing the server.

**Example:**

```py
def on_server_will_close(self, server):
    logging.info("[Server #%s] Closing the server.", os.getpid())
```
````
