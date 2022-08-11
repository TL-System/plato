# Clients

## Customizing clients using inheritance

The common practice is to customize the client using inheritance for important features that change internal states within a client. To customize the client using inheritance, subclass the `simple.Client` class in `plato.clients`, and override the following methods.

```{admonition} **customize_report(self, report) -> SimpleNamespace**
Overide this method to customize a client's report to be sent to the server.

**Example:**

```py
def customize_report(self, report) -> SimpleNamespace:
    loss = self.get_loss()
    report.valuation = self.calc_valuation(report.num_samples, loss)
    return report
```

## Customizing clients using callbacks

For infrastructure changes, such as logging and recording metrics, we tend to customize the local training process using callbacks instead. The advantage of using callbacks is that one can pass a list of multiple callbacks to the client when it is initialized, and they will be called in their order in the provided list. This helps when it is necessary to group features into different callback classes.

Within the implementation of these callback methods, one can access additional information about the local training by using the `client` instance. 

To use callbacks, subclass the `ClientCallback` class in `plato.callbacks.client`, and override the following methods:

