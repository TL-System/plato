"""
FedTP learns a personalized self-attention layer for each client
while the parameters of the other layers are shared among the clients.

Reference:
Li, Hongxia, Zhongyi Cai, Jingya Wang, Jiangnan Tang, Weiping Ding, Chin-Teng Lin, and Ye Shi.
"FedTP: Federated Learning by Transformer Personalization."
arXiv preprint arXiv:2211.01572 (2022).

https://arxiv.org/pdf/2211.01572v1.pdf.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional, cast

import fedtp_algorithm
import hypernetworks
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from plato.config import Config
from plato.servers import fedavg
from plato.servers.strategies.aggregation import FedAvgAggregationStrategy


class FedTPAggregationStrategy(FedAvgAggregationStrategy):
    """Aggregation strategy that delegates to the FedTP server logic."""

    async def aggregate_weights(  # pylint: disable=arguments-differ
        self, updates, baseline_weights, weights_received, context
    ):
        server = getattr(context, "server", None)
        if server is None or not hasattr(server, "_aggregate_weights"):
            return None
        return await server._aggregate_weights(
            updates, baseline_weights, weights_received
        )


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
        super().__init__(
            model,
            datasource,
            algorithm,
            trainer,
            callbacks,
            aggregation_strategy=FedTPAggregationStrategy(),
        )
        self.hnet: Module = hypernetworks.ViTHyper(
            Config().clients.total_clients,
            Config().parameters.hypernet.embed_dim,
            Config().parameters.hypernet.hidden_dim,
            Config().parameters.hypernet.dim,
            heads=Config().parameters.hypernet.num_heads,
            dim_head=Config().parameters.hypernet.dim_head,
            n_hidden=3,
            depth=Config().parameters.hypernet.depth,
            client_sample=Config().clients.per_round,
        ).to(Config().device())
        self.hnet_optimizer: Optimizer | None = None
        self.attentions: Dict[int, OrderedDict[str, Tensor]] = {}
        self.current_attention: Optional[OrderedDict[str, Tensor]] = None

    def training_will_start(self) -> None:
        """Assign optimizer particular for hypernetwork."""
        algorithm = cast(fedtp_algorithm.ServerAlgorithm, self.require_algorithm())
        self.hnet_optimizer = algorithm.get_hnet_optimizer(self.hnet)
        super().training_will_start()

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        """Generate personalized attention for models of each client and keep a copy on server."""
        algorithm = cast(fedtp_algorithm.ServerAlgorithm, self.require_algorithm())
        attentions_customized = algorithm.generate_attention(self.hnet, client_id)
        if not isinstance(attentions_customized, OrderedDict):
            raise RuntimeError(
                "FedTP hypernetwork must return an OrderedDict of weights."
            )
        self.attentions[client_id] = attentions_customized
        self.current_attention = attentions_customized
        return super().customize_server_response(
            server_response=server_response, client_id=client_id
        )

    def customize_server_payload(self, payload):
        """Change the attention in payload into personalized attention."""
        payload = super().customize_server_payload(payload)
        if self.current_attention is None:
            return payload
        for weight_name, tensor in self.current_attention.items():
            if weight_name in payload:
                payload[weight_name].copy_(tensor)
        return payload

    async def _aggregate_weights(self, updates, baseline_weights, weights_received):
        """Aggregation of weights in FedTP."""
        algorithm = cast(fedtp_algorithm.ServerAlgorithm, self.require_algorithm())
        deltas_received = algorithm.compute_weight_deltas(
            baseline_weights, weights_received
        )

        self.total_samples = sum(update.report.num_samples for update in updates)

        grads_update: list[Tensor] = []
        for idx, update in enumerate(updates):
            node_weights = self.attentions.get(update.client_id)
            if node_weights is None:
                raise RuntimeError(
                    f"Missing attention weights for client {update.client_id}."
                )
            delta_theta = OrderedDict(
                {
                    key: node_weights[key] - weights_received[idx][key]
                    for key in node_weights.keys()
                }
            )
            hnet_grads = algorithm.calculate_hnet_grads(
                node_weights, delta_theta, self.hnet
            )

            contribution = update.report.num_samples / self.total_samples
            if idx == 0:
                grads_update = [contribution * grad for grad in hnet_grads]
            else:
                for grad_idx, hnet_grad in enumerate(hnet_grads):
                    grads_update[grad_idx] += contribution * hnet_grad

        if self.hnet_optimizer is None:
            raise RuntimeError("Hypernetwork optimizer has not been initialised.")

        self.hnet_optimizer.zero_grad()

        for param, grad in zip(self.hnet.parameters(), grads_update):
            param.grad = grad
        self.hnet_optimizer.step()

        aggregated_deltas = await super().aggregate_deltas(updates, deltas_received)
        aggregated_weights = OrderedDict()
        for name, weight in baseline_weights.items():
            aggregated_weights[name] = weight + aggregated_deltas[name]
        return aggregated_weights
