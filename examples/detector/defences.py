import logging
from collections import OrderedDict

import numpy as np
import torch

from plato.config import Config

registered_defences = {}


def get():
    defence_type = (
        Config().server.defence_type
        if hasattr(Config().server, "defence_type")
        else None
    )

    if defence_type is None:
        logging.info("No defence is applied.")
        return lambda x: x

    if defence_type in registered_defences:
        registered_defence = registered_defences[defence_type]
        logging.info(f"Clients perform {defence_type} defence.")
        return registered_defence

    raise ValueError(f"No such defence: {defence_type}")


def flatten_weights(weights):
    flattened_weights = []

    for weight in weights:
        flattened_weight = []
        for name in weight.keys():
            flattened_weight = (
                weight[name].view(-1)
                if not len(flattened_weight)
                else torch.cat((flattened_weight, weight[name].view(-1)))
            )

        flattened_weights = (
            flattened_weight[None, :]
            if not len(flattened_weights)
            else torch.cat((flattened_weights, flattened_weight[None, :]), 0)
        )
    return flattened_weights


def median(weights_attacked):
    """Aggregate weight updates from the clients using median."""
    """

        deltas_received = self.compute_weight_deltas(updates)
        reports = [report for (__, report, __, __) in updates]
        clients_id = [client for (client, __, __, __) in updates]

        # The number of malicious clients is known to the server.
        # This setting seems unreasonable
        n_attackers = 0
        for client_id in clients_id:
            if client_id <= self.n_attackers:
                n_attackers = n_attackers + 1

        name_list = []
        for name, delta in deltas_received[0].items():
            name_list.append(name)

        all_deltas_vector = []
        for i, delta_received in enumerate(deltas_received):
            delta_vector = []
            for name in name_list:
                delta_vector = (
                    delta_received[name].view(-1)
                    if not len(delta_vector)
                    else torch.cat((delta_vector, delta_received[name].view(-1)))
                )
            all_deltas_vector = (
                delta_vector[None, :]
                if not len(all_deltas_vector)
                else torch.cat((all_deltas_vector, delta_vector[None, :]), 0)
            )

        n_clients = all_deltas_vector.shape[0]
        logging.info("[%s] n_clients: %d", self, n_clients)
        logging.info("[%s] n_attackers: %d", self, n_attackers)
        """
    weights_attacked = flatten_weights(weights_attacked)

    median_delta_vector = torch.median(weights_attacked, dim=0)[0]
    # name list?
    # median_update = {
    #    name: self.trainer.zeros(weights.shape)
    #    for name, weights in deltas_received[0].items()
    # }

    start_index = 0
    median_update = OrderedDict()

    for weight in weights_attacked:
        for name in weight.keys():
            median_update[name] = median_delta_vector[
                start_index : start_index + len(median_update[name].view(-1))
            ].reshape(median_update[name].shape)
        start_index = start_index + len(median_update[name].view(-1))

    # for name in name_list:
    # median_update[name] = median_delta_vector[
    # start_index : start_index + len(median_update[name].view(-1))
    # ].reshape(median_update[name].shape)
    # start_index = start_index + len(median_update[name].view(-1))
    logging.info(f"Finished Median server aggregation.")
    return median_update
