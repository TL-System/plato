# Clients

## Customizing clients using inheritance

The common practice is to customize the client using inheritance for important features that change the state of the local training process. To customize the client using inheritance, subclass the `simple.Client` class in `plato.clients`, and override methods.

```{admonition} **client_train_end(self)**
Overide this method to complete additional tasks after the local training ends.
```

## Customizing clients using callbacks

For infrastructure changes, such as logging and recording metrics, we tend to customize the local training process using callbacks instead. The advantage of using callbacks is that one can pass a list of multiple callbacks to the client when it is initialized, and they will be called in their order in the provided list. This helps when it is necessary to group features into different callback classes.

Within the implementation of these callback methods, one can access additional information about the local training by using the `client` instance. 

To use callbacks, subclass the `ClientCallback` class in `plato.callbacks.client`, and override the following methods:

```{admonition} **on_client_train_end(self, client)**
Overide this method to complete additional tasks after the local training ends.
```
