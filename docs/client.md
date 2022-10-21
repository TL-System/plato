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

```{admonition} **inbound_received(self, inbound_processor)**
Override this method to complete additional tasks before the inbound processors start to process the data received from the server.

`inbound_processor` the pipeline of inbound processors. The list of inbound processor instances can be accessed through its attribute 'processors', as in the following example.

**Example:**

```py
def inbound_received(self, inbound_processor):
    # insert a customized processor to the list of inbound processors
    customized_processor = DummyProcessor(
            client_id=client.client_id,
            current_round=client.current_round,
            name="DummyProcessor",
        )

    inbound_processor.processors.insert(0, customized_processor) 
```

```{admonition} **inbound_processed(self, data)**
Override this method to conduct customized operations to generate client's response to server when inbound data from server has been processed. Default training function will be called if left undefined.

`data` the inbound data after being processed by inbound processors, e.g., model weights before loaded to the trainer.

**Example:**

```py
async def inbound_processed(self, data: Any) -> (SimpleNamespace, Any):
    report, payload = await self.customized_train(data)
    return report, payload
```

```{admonition} **outbound_ready(self, report, outbound_processor)**
Override this method to complete additional tasks before the outbound processors start to process the data to be sent to the server.

`report` the metadata sent back to the server, e.g., training time, accuracy, etc.

`outbound_processor` the pipeline of outbound processors. The list of inbound processor instances can be accessed through its attribute 'processors', as in the following example.

**Example:**

```py
def outbound_ready(self, report, outbound_processor):
    # customize report 
    loss = self.get_loss()
    report.valuation = self.calc_valuation(report.num_samples, loss)
    
    # remove the first processor from the list of outbound processors
    outbound_processor.processors.pop() 
```

<!-- ```{admonition} **customize_report(self, report)**
Override this method to customize a client's report with additional information.

**Example:**

```py
def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
    loss = self.get_loss()
    report.valuation = self.calc_valuation(report.num_samples, loss)
    return report
``` -->

## Customizing clients using callbacks

For infrastructure changes, such as logging and recording metrics, we tend to customize the client using callbacks instead. The advantage of using callbacks is that one can pass a list of multiple callbacks to the client when it is initialized, and they will be called in their order in the provided list. This helps when it is necessary to group features into different callback classes.

Within the implementation of these callback methods, one can access additional information about the local training by using the `client` instance. 

To use callbacks, subclass the `ClientCallback` class in `plato.callbacks.client`, and override the following methods:


````{admonition} **on_inbound_received(self, client, inbound_processor)**
Override this method to complete additional tasks before the inbound processors start to process the data received from the server.

`inbound_processor` the pipeline of inbound processors. The list of inbound processor instances can be accessed through its attribute 'processors'.

<!-- 
**Example:**

```py
def on_inbound_received(self, client, inbound_processor):
    # insert a customized processor to the list of inbound processors
    customized_processor = DummyProcessor(
            client_id=client.client_id,
            current_round=client.current_round,
            name="DummyProcessor",
        )

    inbound_processor.processors.insert(0, customized_processor) 
``` -->
````


````{admonition} **on_inbound_processed(self, client, data)**
Override this method to complete additional tasks when the inbound data has been processed by inbound processors.

`data` the inbound data after being processed by inbound processors, e.g., model weights before loaded to the trainer.

**Example:**

```py
def on_inbound_processed(self, client, data):
    # print the layer names of the model weights before further operations
    for name, weights in data:
        print(name)
```
````

````{admonition} **on_outbound_ready(self, client, outbound_processor)**
Override this method to complete additional tasks before the outbound processors start to process the data to be sent to the server.

`outbound_processor` the pipeline of outbound processors. The list of inbound processor instances can be accessed through its attribute 'processors'.
<!-- 
**Example:**

```py
def on_outbound_ready(self, client, outbound_processor):
    # remove the first processor from the list of outbound processors
    outbound_processor.processors.pop() 
``` -->
````