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

import hypernetworks
import fedtp_algorithm

from plato.servers import fedavg
from plato.config import Config


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
            dim_head=64,
            n_hidden=3,
            depth=Config().parameters.hypernet.depth,
            client_sample=Config().clients.per_round,
        )
        self.hnet_optimizer = None
        self.attentions = {}
        self.current_attention = None

    def training_will_start(self) -> None:
        self.hnet_optimizer = self.algorithm.get_hnet_optimizer(self.hnet)
        return super().training_will_start()

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        attentions_customized = self.algorithm.generate_attention(self.hnet, client_id)

        self.attentions[client_id] = attentions_customized
        self.current_attention = attentions_customized
        return super().customize_server_response(
            server_response=server_response, client_id=client_id
        )

    def customize_server_payload(self, payload):
        payload = super().customize_server_payload(payload)
        for weight_name in self.current_attention:
            payload[weight_name].copy_(self.current_attention[weight_name])
        return payload

    async def aggregate_deltas(self, updates, deltas_received):
        self.total_samples = sum(update.report.num_samples for update in updates)

        grads_update = OrderedDict()
        for idx, update in enumerate(updates):
            node_weights = self.attentions[update.client_id]
            delta_theta = OrderedDict(
                {k: deltas_received[idx][k] for k in node_weights.keys()}
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
            param.grad = grad
        self.hnet_optimizer.step()

        avg_update = await super().aggregate_deltas(updates, deltas_received)
        return avg_update
