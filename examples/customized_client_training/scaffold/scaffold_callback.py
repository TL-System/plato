"""
Customize the list of inbound and outbound processors for scaffold clients through callbacks.
"""

import logging
from typing import Any, List, Optional

from plato.callbacks.client import ClientCallback
from plato.processors import base
from plato.trainers.strategies.algorithms.scaffold_strategy import (
    validate_control_variates,
)


class ExtractControlVariatesProcessor(base.Processor):
    """
    A processor for clients to extract the control variates that are attached to the payload
    by the server.
    """

    def __init__(self, client_id, trainer, **kwargs) -> None:
        super().__init__(**kwargs)

        self.client_id = client_id
        self.trainer: Optional[Any] = trainer

    def process(self, data: Any) -> Any:
        trainer = self.trainer
        if trainer is None:
            raise ValueError("SCAFFOLD inbound processor requires the active trainer.")
        trainer.additional_data = None
        trainer.context.state.pop("server_control_variate", None)
        trainer.context.state.pop("client_control_variate_delta", None)
        if not isinstance(data, (list, tuple)) or len(data) != 2:
            raise ValueError("SCAFFOLD requires a current [weights, server_controls] payload.")
        controls = validate_control_variates(trainer.model, data[1])
        trainer.context.state["server_control_variate"] = controls
        trainer.additional_data = controls
        return data[0]


class SendControlVariateProcessor(base.Processor):
    """
    A processor for clients to attach additional items to the client payload.
    Sends Δc_i (client control variate delta) as required by SCAFFOLD Eq. (5).
    """

    def __init__(self, client_id, trainer, **kwargs) -> None:
        super().__init__(**kwargs)

        self.client_id = client_id
        self.trainer = trainer

    def process(self, data: Any) -> List[Any]:
        trainer = self.trainer
        if trainer is None:
            raise ValueError("SCAFFOLD outbound processor requires the active trainer.")
        delta = trainer.model_update_strategy.get_update_payload(
            trainer.context
        )["control_variate_delta"]
        return [data, validate_control_variates(trainer.model, delta)]


class ScaffoldCallback(ClientCallback):
    """
    A client callback that dynamically inserts processors into the current list of inbound
    processors.
    """

    def on_inbound_received(self, client, inbound_processor):
        """
        Insert an ExtractPayloadProcessor to the list of inbound processors.
        """
        processors = inbound_processor.processors
        for processor in processors:
            if isinstance(processor, ExtractControlVariatesProcessor):
                processor.client_id = client.client_id
                processor.trainer = client.trainer
                return

        extract_payload_processor = ExtractControlVariatesProcessor(
            client_id=client.client_id,
            trainer=client.trainer,
            name="ExtractControlVariatesProcessor",
        )
        decode_index = next(
            (
                index
                for index, proc in enumerate(processors)
                if getattr(proc, "name", None) == "safetensor_decode"
            ),
            None,
        )
        if decode_index is None:
            processors.append(extract_payload_processor)
        else:
            processors.insert(decode_index + 1, extract_payload_processor)

        logging.info(
            "[%s] List of inbound processors: %s.", client, inbound_processor.processors
        )

    def on_outbound_ready(self, client, report, outbound_processor):
        """
        Insert a SendControlVariateProcessor to the list of outbound processors.
        """
        processors = outbound_processor.processors
        for processor in processors:
            if isinstance(processor, SendControlVariateProcessor):
                processor.client_id = client.client_id
                processor.trainer = client.trainer
                return

        send_payload_processor = SendControlVariateProcessor(
            client_id=client.client_id,
            trainer=client.trainer,
            name="SendControlVariateProcessor",
        )

        encode_index = next(
            (
                index
                for index, proc in enumerate(processors)
                if getattr(proc, "name", None) == "safetensor_encode"
            ),
            None,
        )
        insert_index = encode_index if encode_index is not None else len(processors)
        processors.insert(insert_index, send_payload_processor)

        logging.info(
            "[%s] List of outbound processors: %s.",
            client,
            outbound_processor.processors,
        )
