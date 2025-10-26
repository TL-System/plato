"""
Customize the list of inbound and outbound processors through callbacks.
"""

import logging

from dummy_processor import DummyProcessor

from plato.callbacks.client import ClientCallback


class CustomizeProcessorCallback(ClientCallback):
    """Insert a dummy processor into inbound and outbound pipelines before they run."""

    def _insert_dummy(self, client, processor_pipeline, direction: str) -> None:
        """Prepend the dummy processor if a processor pipeline is available."""
        if processor_pipeline is None or not hasattr(processor_pipeline, "processors"):
            logging.warning(
                "[%s] No %s processor pipeline available; skip customization.",
                client,
                direction,
            )
            return

        logging.info(
            "[%s] Current list of %s processors: %s.",
            client,
            direction,
            processor_pipeline.processors,
        )
        customized_processor = DummyProcessor(
            client_id=client.client_id,
            current_round=client.current_round,
            name="DummyProcessor",
        )
        processor_pipeline.processors.insert(0, customized_processor)

        logging.info(
            "[%s] List of %s processors after modification: %s.",
            client,
            direction,
            processor_pipeline.processors,
        )

    def on_inbound_received(self, client, inbound_processor):
        """
        Insert a dummy processor before inbound processors start to run.
        """
        self._insert_dummy(client, inbound_processor, "inbound")

    def on_outbound_ready(self, client, report, outbound_processor):
        """
        Insert a dummy processor before outbound processors start to run.
        """
        self._insert_dummy(client, outbound_processor, "outbound")
