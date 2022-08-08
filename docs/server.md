# Server

## Customizing server using inheritance

The common practice is to customize the server using inheritance for important features that change the state of the global training process. To customize the server using inheritance, subclass the `fedavg.Server` (or `fedavg_cs.Server` for cross-silo FL) class in `plato.servers`, and override methods.


## Customizing server using callbacks

For infrastructure changes, such as logging and recording metrics, we tend to customize the global training process using callbacks instead. The advantage of using callbacks is that one can pass a list of multiple callbacks to the client when it is initialized, and they will be called in their order in the provided list. This helps when it is necessary to group features into different callback classes.

Within the implementation of these callback methods, one can access additional information about the global training by using the `server` instance. 

To use callbacks, subclass the `ServerCallback` class in `plato.callbacks.server`, and override the following methods:

````{admonition} **on_server_selection_start(self, server)**
Overide this method to complete additional tasks before selecting clients.

**Example:**

```py
def on_server_selection_start(self, server):
    logging.info("[Server #%s] Selecting clients.", os.getpid())
```
````

````{admonition} **on_server_selection_end(self, server)**
Overide this method to complete additional tasks at the end of client selection.
````

````{admonition} **on_server_close_start(self, server)**
Overide this method to complete additional tasks before closing the server.

**Example:**

```py
def on_server_close_start(self, server):
    logging.info("[Server #%s] Closing the server.", os.getpid())
```
````

