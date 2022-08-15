# Clients

## Customizing clients using inheritance

The common practice is to customize the client using inheritance for important features that change internal states within a client. To customize the client using inheritance, subclass the `simple.Client` class (or `edge.Client` for cross-silo federated learning) in `plato.clients`, and override the following methods:

```{admonition} **process_server_response(self, server_response)**
Override this method to conduct additional client-specific processing on the server response.

**Example:**

```py
def process_server_response(self, server_response):
    if "current_global_round" in server_response:
        self.server.current_global_round = server_response["current_global_round"]
```

```{admonition} **customize_report(self, report)**
Override this method to customize a client's report with additional information.

**Example:**

```py
def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
    loss = self.get_loss()
    report.valuation = self.calc_valuation(report.num_samples, loss)
    return report
```

## Customizing clients using callbacks

For infrastructure changes, such as logging and recording metrics, we tend to customize the client using callbacks instead. The advantage of using callbacks is that one can pass a list of multiple callbacks to the client when it is initialized, and they will be called in their order in the provided list. This helps when it is necessary to group features into different callback classes.

Within the implementation of these callback methods, one can access additional information about the local training by using the `client` instance. 

To use callbacks, subclass the `ClientCallback` class in `plato.callbacks.client`, and override the following methods:

