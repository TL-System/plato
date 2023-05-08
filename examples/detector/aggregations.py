import torch
import logging
from plato.config import Config
from scipy.stats import norm
from collections import OrderedDict
import numpy as np

def get():

    aggregation_type = (
        Config().server.secure_aggregation_type
        if hasattr(Config().server, "secure_aggregation_type")
        else None
    )

    if aggregation_type is None:
        logging.info("No secure aggregation is applied.")
        return lambda x: x

    if aggregation_type in registered_aggregations:
        registered_aggregation = registered_aggregations[aggregation_type]
        logging.info(f"Clients perform {aggregation_type} aggregation.")
        return registered_aggregation

    raise ValueError(f"No such aggregation: {aggregation_type}")

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

def median(updates,baseline_weights, weights_attacked):
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
        flattened_weights = flatten_weights(weights_attacked)

        median_delta_vector = torch.median(flattened_weights, dim=0)[0]

        start_index = 0
        median_update = OrderedDict()

        for weight in weights_attacked:
            for name in weight.keys():
                median_update[name] = median_delta_vector[
                start_index : start_index + len(weight[name].view(-1))
            ].reshape(weight[name].shape)
            start_index = start_index + len(weight[name].view(-1))

        logging.info(f"Finished Median server aggregation.")
        return median_update

def bulyan(updates,baseline_weights, weights_attacked):
        """Aggregate weight updates from the clients using bulyan."""
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

        total_clients = Config().clients.total_clients
        num_attackers = len(Config().clients.attacker_ids)

        flattened_weights = flatten_weights(weights_attacked)

        bulyan_cluster = []
        candidate_indices = []
        remaining_deltas_vector = flattened_weights
        all_indices = np.arange(len(weights_attacked))

        while (len(bulyan_cluster) < (total_clients - 2 * num_attackers)) and (
            len(bulyan_cluster) < (total_clients - 2 - num_attackers)
        ):
            distances = []
            for deltas_vector in remaining_deltas_vector:
                distance = (
                    torch.norm((remaining_deltas_vector - deltas_vector), dim=1) ** 2
                )
                distances = (
                    distance[None, :]
                    if not len(distances)
                    else torch.cat((distances, distance[None, :]), 0)
                )

            distances = torch.sort(distances, dim=1)[0]

            scores = torch.sum(
                distances[:, : len(remaining_deltas_vector) - 2 - num_attackers], dim=1
            )
            indices = torch.argsort(scores)[
                : len(remaining_deltas_vector) - 2 - num_attackers
            ]

            candidate_indices.append(all_indices[indices[0].cpu().numpy()])
            all_indices = np.delete(all_indices, indices[0].cpu().numpy())
            bulyan_cluster = (
                remaining_deltas_vector[indices[0]][None, :]
                if not len(bulyan_cluster)
                else torch.cat(
                    (bulyan_cluster, remaining_deltas_vector[indices[0]][None, :]), 0
                )
            )
            remaining_deltas_vector = torch.cat(
                (
                    remaining_deltas_vector[: indices[0]],
                    remaining_deltas_vector[indices[0] + 1 :],
                ),
                0,
            )

        print("dim of bulyan cluster ", bulyan_cluster.shape)

        n, d = bulyan_cluster.shape
        param_median = torch.median(bulyan_cluster, dim=0)[0]
        sort_idx = torch.argsort(torch.abs(bulyan_cluster - param_median), dim=0)
        sorted_deltas_vector = bulyan_cluster[sort_idx, torch.arange(d)[None, :]]

        mean_delta_vector = torch.mean(
            sorted_deltas_vector[: n - 2 * num_attackers], dim=0
        )
        krum_candidate = np.array(candidate_indices) #?

        start_index = 0
        bulyan_update = OrderedDict()

        for weight in weights_attacked:
            for name in weight.keys():
                bulyan_update[name] = mean_delta_vector[
                start_index : start_index + len(weight[name].view(-1))
            ].reshape(weight[name].shape)
            start_index = start_index + len(weight[name].view(-1))
        
        logging.info(f"Finished Bulyan server aggregation.")
        return bulyan_update

def krum(updates,baseline_weights, weights_attacked):
        """Aggregate weight updates from the clients using krum."""
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
        flattened_weights = flatten_weights(weights_attacked)

        candidates = []
        candidate_indices = []
        remaining_deltas_vector = flattened_weights
        all_indices = np.arange(len(flattened_weights))
        while len(candidates) < 1:
            distances = []
            for deltas_vector in remaining_deltas_vector:
                distance = (
                    torch.norm((remaining_deltas_vector - deltas_vector), dim=1) ** 2
                )
                distances = (
                    distance[None, :]
                    if not len(distances)
                    else torch.cat((distances, distance[None, :]), 0)
                )

            distances = torch.sort(distances, dim=1)[0]

            scores = torch.sum(
                distances[:, : len(remaining_deltas_vector) - 2 - n_attackers], dim=1
            )
            indices = torch.argsort(
                scores
            )  # [:len(remaining_deltas_vector) - 2 - n_attackers]

            candidate_indices.append(all_indices[indices[0].cpu().numpy()])
            all_indices = np.delete(all_indices, indices[0].cpu().numpy())
            candidates = (
                remaining_deltas_vector[indices[0]][None, :]
                if not len(candidates)
                else torch.cat(
                    (candidates, remaining_deltas_vector[indices[0]][None, :]), 0
                )
            )
            remaining_deltas_vector = torch.cat(
                (
                    remaining_deltas_vector[: indices[0]],
                    remaining_deltas_vector[indices[0] + 1 :],
                ),
                0,
            )

        mean_delta_vector = torch.mean(candidates, dim=0)

        start_index = 0
        krum_update = OrderedDict()

        for weight in weights_attacked:
            for name in weight.keys():
                krum_update[name] = mean_delta_vector[
                start_index : start_index + len(weight[name].view(-1))
            ].reshape(weight[name].shape)
            start_index = start_index + len(weight[name].view(-1))
        
        logging.info(f"Finished krum server aggregation.")
        return krum_update

        """

        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0].items()
        }

        start_index = 0
        for name in name_list:
            avg_update[name] = mean_delta_vector[
                start_index : start_index + len(avg_update[name].view(-1))
            ].reshape(avg_update[name].shape)
            start_index = start_index + len(avg_update[name].view(-1))
        return avg_update
        """


registered_aggregations = {"Median":median, "Bulyan": bulyan,"Krum":krum}
