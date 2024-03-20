import logging
from plato.callbacks.client import ClientCallback

class SetupPseudoLabelCallback(ClientCallback):
    def on_inbound_received(self, client, inbound_processor):
        logging.info(f"[{client}] Read pseudo labels from file.")
        client.datasource.setup_client_datasource(client.client_id)

class EvalPseudoLabelCallback(ClientCallback):
    def on_outbound_ready(self, client, report, outbound_processor):
        logging.info(f"[{client}] Evaluate predicted pseudo labels.")
        client_id = client.client_id
        client_indices = client.sampler.subset_indices
        client.datasource.eval_pseudo_acc(client_id, client_indices)