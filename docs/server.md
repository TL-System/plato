# Servers

## Customizing servers using inheritance

The common practice is to customize the server using inheritance for important features that change the state of the server. To customize the server using inheritance, subclass the `fedavg.Server` (or `fedavg_cs.Server` for cross-silo federated learning) class in `plato.servers`, and override the following methods:


````{admonition} **weights_received(self, weights_received)**
Overide this method to complete additional tasks after the updated weights have been received.

`weights_received` the updated weights that have been received from the clients.

**Example:**

```py
def weights_received(self, weights_received):
    """
    Event called after the updated weights have been received.
    """
    self.control_variates_received = [weight[1] for weight in weights_received]
```
````

````{admonition} **weights_aggregated(self, updates)**
Overide this method to complete additional tasks after aggregating weights.

`updates` the client updates received at the server.
````

## Customizing servers using callbacks

For infrastructure changes, such as logging and recording metrics, we tend to customize the global training process using callbacks instead. The advantage of using callbacks is that one can pass a list of multiple callbacks to the client when it is initialized, and they will be called in their order in the provided list. This helps when it is necessary to group features into different callback classes.

Within the implementation of these callback methods, one can access additional information about the global training by using the `server` instance. 

To use callbacks, subclass the `ServerCallback` class in `plato.callbacks.server`, and override the following methods:

````{admonition} **on_weights_received(self, server, weights_received)**
Overide this method to complete additional tasks after the updated weights have been received.

`weights_received` the updated weights that have been received from the clients.
````

````{admonition} **on_weights_aggregated(self, server, updates)**
Overide this method to complete additional tasks after aggregating weights.

`updates` the client updates received at the server.

**Example:**

```py
def on_weights_aggregated(self, server, updates):
    logging.info("[Server #%s] Finished aggregating weights.", os.getpid())
```
````

````{admonition} **on_server_will_close(self, server)**
Overide this method to complete additional tasks before closing the server.

**Example:**

```py
def on_server_will_close(self, server):
    logging.info("[Server #%s] Closing the server.", os.getpid())
```
````

