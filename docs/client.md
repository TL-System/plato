# Clients

## Customizing clients using inheritance

The common practice is to customize the client using inheritance for important features that change internal states within a client. To customize the client using inheritance, subclass the `simple.Client` class (or `edge.Client` for cross-silo federated learning) in `plato.clients`, and override the following methods:

```{admonition} **configure()**
**`def configure(self) -> None`**

Override this method to implement additional tasks for initializing and configuring the client. Make sure that `super().configure()` is called first.
```

````{admonition} **process_server_response()**
**`def process_server_response(self, server_response) -> None`**

Override this method to conduct additional client-specific processing on the server response.

**Example:**

```py
def process_server_response(self, server_response):
    if "current_global_round" in server_response:
        self.server.current_global_round = server_response["current_global_round"]
```
````

````{admonition} **inbound_received()**
**`def inbound_received(self, inbound_processor)`**

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
````

````{admonition} **inbound_processed()**
**`def inbound_processed(self, processed_inbound_payload)`**

Override this method to conduct customized operations to generate a client's response to the server when inbound data from the server has been processed.

`processed_inbound_payload` the inbound payload after being processed by inbound processors, e.g., model weights before loaded to the trainer.

**Returns:** the report and the outbound payload.

**Example:**

```py
async def inbound_processed(self, processed_inbound_payload: Any) -> (SimpleNamespace, Any):
    report, outbound_payload = await self.customized_train(processed_inbound_payload)
    return report, outbound_payload
```
````

````{admonition} **outbound_ready()**
**`def outbound_ready(self, report, outbound_processor)`**

Override this method to complete additional tasks before the outbound processors start to process the data to be sent to the server.

`report` the metadata sent back to the server, e.g., training time, accuracy, etc.

`outbound_processor` the pipeline of outbound processors. The list of inbound processor instances can be accessed through its attribute 'processors', as in the following example.

**Example:**

```py
def outbound_ready(self, report, outbound_processor):
    # customize the report 
    loss = self.get_loss()
    report.valuation = self.calc_valuation(report.num_samples, loss)
    
    # remove the first processor from the list of outbound processors
    outbound_processor.processors.pop() 
```
````

## Customizing clients using callbacks

For infrastructure changes, such as logging and recording metrics, we tend to customize the client using callbacks instead. The advantage of using callbacks is that one can pass a list of multiple callbacks to the client when it is initialized, and they will be called in their order in the provided list. This helps when it is necessary to group features into different callback classes.

Within the implementation of these callback methods, one can access additional information about the local training by using the `client` instance. 

To use callbacks, subclass the `ClientCallback` class in `plato.callbacks.client`, and override the following methods, then pass it to the client when it is initialized, or call `client.add_callbacks` after initialization. Examples can be found in `examples/callbacks`.


````{admonition} **on_inbound_received()**
**`def on_inbound_received(self, client, inbound_processor)`**

Override this method to complete additional tasks before the inbound processors start to process the data received from the server.

`inbound_processor` the pipeline of inbound processors. The list of inbound processor instances can be accessed through its attribute 'processors'.
````


````{admonition} **on_inbound_processed()**
**`def on_inbound_processed(self, client, data)`**

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

````{admonition} **on_outbound_ready()**
**`def on_outbound_ready(self, client, outbound_processor)`**

Override this method to complete additional tasks before the outbound processors start to process the data to be sent to the server.

`outbound_processor` the pipeline of outbound processors. The list of inbound processor instances can be accessed through its attribute 'processors'.
````