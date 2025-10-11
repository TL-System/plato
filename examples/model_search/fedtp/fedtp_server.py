"""
FedTP learns a personalized self-attention layer for each client
while the parameters of the other layers are shared among the clients.

Reference:
Li, Hongxia, Zhongyi Cai, Jingya Wang, Jiangnan Tang, Weiping Ding, Chin-Teng Lin, and Ye Shi.
"FedTP: Federated Learning by Transformer Personalization."
arXiv preprint arXiv:2211.01572 (2022).

https://arxiv.org/pdf/2211.01572v1.pdf.
"""

from collections import OrderedDict

import fedtp_algorithm
import hypernetworks

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """The federated learning server for the FedTP."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=fedtp_algorithm.ServerAlgorithm,
        trainer=None,
        callbacks=None,
    ):
        # pylint:disable=too-many-arguments
        super().__init__(model, datasource, algorithm, trainer, callbacks)
        self.hnet = hypernetworks.ViTHyper(
            Config().clients.total_clients,
            Config().parameters.hypernet.embed_dim,
            Config().parameters.hypernet.hidden_dim,
            Config().parameters.hypernet.dim,
            heads=Config().parameters.hypernet.num_heads,
            dim_head=Config().parameters.hypernet.dim_head,
            n_hidden=3,
            depth=Config().parameters.hypernet.depth,
            client_sample=Config().clients.per_round,
        )
        self.hnet = self.hnet.to(Config().device())
        self.hnet_optimizer = None
        self.attentions = {}
        self.current_attention = None

    def training_will_start(self) -> None:
        """Assign optimizer particular for hypernetwork."""
        self.hnet_optimizer = self.algorithm.get_hnet_optimizer(self.hnet)
        return super().training_will_start()

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        """Generate personalized attention for models of each client and have a copy on server."""
        attentions_customized = self.algorithm.generate_attention(self.hnet, client_id)

        self.attentions[client_id] = attentions_customized
        self.current_attention = attentions_customized
        return super().customize_server_response(
            server_response=server_response, client_id=client_id
        )

    def customize_server_payload(self, payload):
        """Change the attention in payload into personalized attention."""
        payload = super().customize_server_payload(payload)
        for weight_name in self.current_attention:
            payload[weight_name].copy_(self.current_attention[weight_name])
        return payload

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Aggregation of weights in FedTP."""

        # pylint:disable=too-many-locals

        deltas_recieved = self.algorithm.compute_weight_deltas(
            baseline_weights, weights_received
        )

        self.total_samples = sum(update.report.num_samples for update in updates)

        grads_update = OrderedDict()
        for idx, update in enumerate(updates):
            node_weights = self.attentions[update.client_id]
            delta_theta = OrderedDict(
                {
                    k: node_weights[k] - weights_received[idx][k]
                    for k in node_weights.keys()
                }
            )
            hnet_grads = self.algorithm.calculate_hnet_grads(
                node_weights, delta_theta, self.hnet
            )

            if idx == 0:
                grads_update = [
                    update.report.num_samples / self.total_samples * x
                    for x in hnet_grads
                ]
            else:
                for grad_idx, hnet_grad in enumerate(hnet_grads):
                    grads_update[grad_idx] += (
                        update.report.num_samples / self.total_samples * hnet_grad
                    )

        self.hnet_optimizer.zero_grad()

        for param, grad in zip(self.hnet.parameters(), grads_update):
            param.grad = grad  # .copy_(grad)
        self.hnet_optimizer.step()

        avg_update = await super().aggregate_deltas(updates, deltas_recieved)
        return avg_update
