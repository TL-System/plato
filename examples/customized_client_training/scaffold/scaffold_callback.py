"""
Customize the list of inbound and outbound processors for scaffold clients through callbacks.
"""

import logging
import os
import pickle
from typing import Any, List, Optional

from plato.callbacks.client import ClientCallback
from plato.processors import base


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
        if not isinstance(data, list):
            if self.trainer is not None:
                setattr(self.trainer, "additional_data", None)
            return data

        if len(data) > 1 and self.trainer is not None:
            setattr(self.trainer, "additional_data", data[1])

            logging.info(
                "[Client #%d] Control variates extracted from the payload.",
                self.client_id,
            )
            return data[0]

        if self.trainer is not None:
            setattr(self.trainer, "additional_data", None)
        return data[0] if data else None


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
        delta = getattr(self.trainer, "client_control_variate_delta", None)
        data = [data, delta]

        if delta is not None:
            logging.info(
                "[Client #%d] Control variate deltas were attached to the payload.",
                self.client_id,
            )
        else:
            logging.info(
                "[Client #%d] No control variate deltas available; sending None.",
                self.client_id,
            )

        return data


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
        if any(
            isinstance(proc, ExtractControlVariatesProcessor) for proc in processors
        ):
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
        if any(isinstance(proc, SendControlVariateProcessor) for proc in processors):
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
