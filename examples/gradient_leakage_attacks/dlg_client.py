"""A federated learning client for gradient leakage attacks."""

import pickle

from plato.clients import simple
from plato.config import Config


class Client(simple.Client):
    """A federated learning client for gradient leakage attacks."""

    async def inbound_processed(self, processed_inbound_payload):
        """Add extra information along with the original payload for attack validation at server."""
        report, outbound_payload = await super().inbound_processed(
            processed_inbound_payload
        )
        file_path = f"{Config().params['model_path']}/{self.client_id}.pickle"
        with open(file_path, "rb") as handle:
            gt_data, gt_labels, target_grad = pickle.load(handle)

        outbound_payload = (outbound_payload, gt_data, gt_labels, target_grad)
        return report, outbound_payload
