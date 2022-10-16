""" 
Customize the inbound and outbound processor list through callbacks. 
"""

from plato.callbacks.client import ClientCallback

from dummy_processor import DummyProcessor


class CustomizeProcessorCallback(ClientCallback):
    """
    A client callback that dynamically inserts a dummy processor to the existing processor list.
    """

    def on_inbound_process(self, client, inbound_processor):
        """
        Insert a dummy processor to the inbound processor list.
        """
        customized_processor = DummyProcessor(client.client_id, client.current_round)
        inbound_processor.processors.insert(0, customized_processor)

    def on_outbound_process(self, client, outbound_processor):
        """
        Insert a dummy processor to the outbound processor list.
        """
        customized_processor = DummyProcessor(client.client_id, client.current_round)
        outbound_processor.processors.insert(0, customized_processor)
