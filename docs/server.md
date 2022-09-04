# Servers

## Customizing servers using inheritance

The common practice is to customize the server using inheritance for important features that change the state of the server. To customize the server using inheritance, subclass the `fedavg.Server` (or `fedavg_cs.Server` for cross-silo federated learning) class in `plato.servers`, and override the following methods:

```{admonition} **choose_clients(self, clients_pool, clients_count)**
Override this method to implement a customized client selection algorithm, choosing a subset of clients from the client pool.

`clients_pool` a list of available clients for selection.

`clients_count` the number of clients that need to be selected in this round.
```

````{admonition} **weights_received(self, weights_received)**
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

```{admonition} **aggregate_deltas(self, updates, deltas_received)**
In most cases, it is more convenient to aggregate the model deltas from the clients, because this can be performed in a framework-agnostic fashion. Override this method to aggregate the deltas received. This method is needed if `aggregate_weights()` (below) is not defined.

`updates` the client updates received at the server.

`deltas_received` the weight deltas received from the clients.
```

```{admonition} **aggregate_weights(self, updates, baseline_weights, weights_received)**
Sometimes it is more convenient to aggregate the received model weights directly to the global model. In this case, override this method to aggregate the weights received directly to baseline weights. This method is optional, and the server will call this method rather than `aggregate_deltas` when it is defined. Refer to `examples/fedasync/fedasync_server.py` for an example.

`updates` the client updates received at the server.

`baseline_weights` the current weights in the global model.

`weights_received` the weights received from the clients.
```

````{admonition} **weights_aggregated(self, updates)**
Override this method to complete additional tasks after aggregating weights.

`updates` the client updates received at the server.
````

````{admonition} **customize_server_response(self, server_response: dict)**
Override this method to customize the server response with any additional information.

**Example:**

```py
def customize_server_response(self, server_response: dict) -> dict:
    """
    Customizes the server response with any additional information.
    """
    server_response["pruning_amount"] = self.pruning_amount_list
    return server_response
```
````

```{admonition} **customize_server_payload(self, payload)**
Override this method to customize the server payload before sending it to the clients.
```

```{admonition} **server_will_close(self)**
Override this method to complete additional tasks before closing the server.
```

## Customizing servers using callbacks

For infrastructure changes, such as logging and recording metrics, we tend to customize the global training process using callbacks instead. The advantage of using callbacks is that one can pass a list of multiple callbacks to the client when it is initialized, and they will be called in their order in the provided list. This helps when it is necessary to group features into different callback classes.

Within the implementation of these callback methods, one can access additional information about the global training by using the `server` instance. 

To use callbacks, subclass the `ServerCallback` class in `plato.callbacks.server`, and override the following methods:

````{admonition} **on_weights_received(self, server, weights_received)**
Override this method to complete additional tasks after the updated weights have been received.

`weights_received` the updated weights that have been received from the clients.
````

````{admonition} **on_weights_aggregated(self, server, updates)**
Override this method to complete additional tasks after aggregating weights.

`updates` the client updates received at the server.

**Example:**

```py
def on_weights_aggregated(self, server, updates):
    logging.info("[Server #%s] Finished aggregating weights.", os.getpid())
```
````

````{admonition} **on_server_will_close(self, server)**
Override this method to complete additional tasks before closing the server.

**Example:**

```py
def on_server_will_close(self, server):
    logging.info("[Server #%s] Closing the server.", os.getpid())
```
````

