# Clients

## Customizing clients using inheritance

The common practice is to customize the client using inheritance for important features that change the state of the local training process. To customize the client using inheritance, subclass the `simple.Client` class in `plato.clients`, and override methods.


## Customizing clients using callbacks

For infrastructure changes, such as logging and recording metrics, we tend to customize the local training process using callbacks instead. The advantage of using callbacks is that one can pass a list of multiple callbacks to the client when it is initialized, and they will be called in their order in the provided list. This helps when it is necessary to group features into different callback classes.

Within the implementation of these callback methods, one can access additional information about the local training by using the `client` instance. 

To use callbacks, subclass the `ClientCallback` class in `plato.callbacks.client`, and override the following methods:

````{admonition} **on_client_train_start(self, client)**
Overide this method to complete additional tasks before the local training starts.

**Example:**

```py
def on_client_train_start(self, client)
    logging.info(
        fonts.colourize(
            f"[{client}] Started training in communication round #{client.current_round}."
        )
    )
```
````

````{admonition} **on_client_train_end(self, client)**
Overide this method to complete additional tasks after the local training ends.

````
