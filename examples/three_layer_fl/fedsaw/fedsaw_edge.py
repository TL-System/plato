"""
A federated learning client at the edge server in a cross-silo training workload.
"""

import copy
import logging
from collections import OrderedDict

import torch
from torch.nn.utils import prune

from plato.clients import edge
from plato.clients.strategies.base import ClientContext
from plato.clients.strategies.edge import EdgeLifecycleStrategy, EdgeTrainingStrategy
from plato.config import Config


class FedSawEdgeLifecycleStrategy(EdgeLifecycleStrategy):
    """Lifecycle strategy that records pruning amounts for FedSaw edge clients."""

    def process_server_response(self, context, server_response):
        super().process_server_response(context, server_response)

        pruning_amounts = server_response.get("pruning_amount")
        if pruning_amounts is None:
            return

        logical_client_id = Config().args.id
        try:
            pruning_amount = pruning_amounts[str(logical_client_id)]
        except (KeyError, TypeError):
            return

        owner = context.owner
        if owner is not None and hasattr(owner, "server"):
            owner.server.edge_pruning_amount = pruning_amount


class FedSawEdgeTrainingStrategy(EdgeTrainingStrategy):
    """Training strategy that prunes aggregated updates on the edge server."""

    async def train(self, context: ClientContext):
        owner = context.owner
        if owner is None:
            raise RuntimeError("FedSaw edge strategy requires an owning client.")

        server = owner.server
        previous_weights = copy.deepcopy(server.algorithm.extract_weights())

        report, new_weights = await super().train(context)

        weight_updates = self._prune_updates(context, previous_weights, new_weights)
        logging.info(
            "[Edge Server #%d] Pruned its aggregated updates.", context.client_id
        )

        return report, weight_updates

    def _prune_updates(self, context, previous_weights, new_weights):
        owner = context.owner
        server = owner.server

        updates = self._compute_weight_updates(previous_weights, new_weights)
        server.algorithm.load_weights(updates)
        updates_model = server.algorithm.model

        parameters_to_prune = []
        for _, module in updates_model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                parameters_to_prune.append((module, "weight"))

        if (
            hasattr(Config().clients, "pruning_method")
            and Config().clients.pruning_method == "random"
        ):
            pruning_method = prune.RandomUnstructured
        else:
            pruning_method = prune.L1Unstructured

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=pruning_method,
            amount=server.edge_pruning_amount,
        )

        for module, name in parameters_to_prune:
            prune.remove(module, name)

        return updates_model.cpu().state_dict()

    @staticmethod
    def _compute_weight_updates(previous_weights, new_weights):
        deltas = OrderedDict()
        for name, new_weight in new_weights.items():
            previous_weight = previous_weights[name]
            deltas[name] = new_weight - previous_weight
        return deltas


class Client(edge.Client):
    """A federated learning client at the edge server in a cross-silo training workload."""

    def __init__(
        self,
        server,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
    ):
        super().__init__(
            server=server,
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        payload_strategy = self.payload_strategy
        reporting_strategy = self.reporting_strategy
        communication_strategy = self.communication_strategy

        self._configure_composable(
            lifecycle_strategy=FedSawEdgeLifecycleStrategy(),
            payload_strategy=payload_strategy,
            training_strategy=FedSawEdgeTrainingStrategy(),
            reporting_strategy=reporting_strategy,
            communication_strategy=communication_strategy,
        )
