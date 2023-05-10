import torch
import logging
from plato.config import Config
from scipy.stats import norm
from collections import OrderedDict
import numpy as np
import copy


def get():

    aggregation_type = (
        Config().server.secure_aggregation_type
        if hasattr(Config().server, "secure_aggregation_type")
        else None
    )

    if aggregation_type is None:
        logging.info("No secure aggregation is applied.")
        return lambda x, y, z: (x, y, z)

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


def median(updates, baseline_weights, weights_attacked):
    """Aggregate weight updates from the clients using median."""

    flattened_weights = flatten_weights(weights_attacked)

    median_delta_vector = torch.median(flattened_weights, dim=0)[0]

    start_index = 0
    for weight in weights_attacked:  # should iterate only once
        median_update = OrderedDict()
        for name in weight.keys():
            median_update[name] = median_delta_vector[
                start_index : start_index + len(weight[name].view(-1))
            ].reshape(weight[name].shape)
        start_index = start_index + len(weight[name].view(-1))

    logging.info(f"Finished Median server aggregation.")

    return median_update


def bulyan(updates, baseline_weights, weights_attacked):
    """Aggregate weight updates from the clients using bulyan."""

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
            distance = torch.norm((remaining_deltas_vector - deltas_vector), dim=1) ** 2
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

    mean_delta_vector = torch.mean(sorted_deltas_vector[: n - 2 * num_attackers], dim=0)
    krum_candidate = np.array(candidate_indices)  # ?

    start_index = 0
    for weight in weights_attacked:  # should iterate only once
        bulyan_update = OrderedDict()
        for name in weight.keys():
            bulyan_update[name] = mean_delta_vector[
                start_index : start_index + len(weight[name].view(-1))
            ].reshape(weight[name].shape)
        start_index = start_index + len(weight[name].view(-1))

    logging.info(f"Finished Bulyan server aggregation.")
    return bulyan_update


def krum(updates, baseline_weights, weights_attacked):
    """Aggregate weight updates from the clients using krum."""

    remaining_weights = flatten_weights(weights_attacked)

    candidates_weights = []

    while len(candidates_weights) < 1:
        distances = []
        for deltas_vector in remaining_weights:
            distance = torch.norm((remaining_weights - deltas_vector), dim=1) ** 2
            distances = (
                distance[None, :]
                if not len(distances)
                else torch.cat((distances, distance[None, :]), 0)
            )

        distances = torch.sort(distances, dim=1)[0]

        num_attackers_selected = 2  # ?
        scores = torch.sum(
            distances[:, : len(remaining_weights) - 2 - num_attackers_selected], dim=1
        )
        sorted_scores = torch.argsort(scores)
        # put candidates' weights in to candidates (top1)
        candidates_weights = (
            remaining_weights[sorted_scores[0]][None, :]
            if not len(candidates_weights)
            else torch.cat(
                (candidates_weights, remaining_weights[sorted_scores[0]][None, :]), 0
            )
        )
        remaining_weights = torch.cat(
            (
                remaining_weights[: sorted_scores[0]],
                remaining_weights[sorted_scores[0] + 1 :],
            ),
            0,
        )
    mean_delta_vector = torch.mean(candidates_weights, dim=0)

    start_index = 0
    krum_update = OrderedDict()

    for weight in weights_attacked:  # should iterate only once
        median_update = OrderedDict()
        for name in weight.keys():
            median_update[name] = mean_delta_vector[
                start_index : start_index + len(weight[name].view(-1))
            ].reshape(weight[name].shape)
        start_index = start_index + len(weight[name].view(-1))

    logging.info(f"Finished krum server aggregation.")
    return krum_update


def trimmed_mean(updates, baseline_weights, weights_attacked):
    """Aggregate weight updates from the clients using trimmed-mean."""
    flattened_weights = flatten_weights(weights_attacked)
    num_attackers = 2  # ?

    n, d = flattened_weights.shape
    param_median = torch.median(flattened_weights, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(flattened_weights - param_median), dim=0)
    sorted_deltas_vector = flattened_weights[sort_idx, torch.arange(d)[None, :]]

    mean_delta_vector = torch.mean(sorted_deltas_vector[: n - 2 * num_attackers], dim=0)

    start_index = 0

    for weight in weights_attacked:  # should iterate only once
        median_update = OrderedDict()
        for name in weight.keys():
            median_update[name] = mean_delta_vector[
                start_index : start_index + len(weight[name].view(-1))
            ].reshape(weight[name].shape)
        start_index = start_index + len(weight[name].view(-1))

    logging.info(f"Finished Trimmed mean server aggregation.")


def AFA_index_finder(input_delta, retrive_all_delta):
    counter = 0
    for curr_delta in retrive_all_delta:
        if input_delta.equal(curr_delta):
            return counter
        counter += 1
    return -1


def afa(updates, baseline_weights, weights_attacked):
    """Aggregate weight updates from the clients using afa."""

    flattened_weights = flatten_weights(weights_attacked)
    clients_id = [client for (client, __, __, __) in updates]

    retrive_flattened_weights = flattened_weights.clone()

    bad_set = []
    remove_set = [1]
    pvalue = {}

    epsilon = 2
    delta_ep = 0.5

    counter = 0
    if not hasattr(self, "good_hist"):
        self.good_hist = np.zeros(Config().clients.total_clients)
        self.bad_hist = np.zeros(Config().clients.total_clients)
        self.alpha0 = 3
        self.beta0 = 3

    for client in clients_id:
        ngood = self.good_hist[client - 1]
        nbad = self.bad_hist[client - 1]
        alpha = self.alpha0 + ngood
        beta = self.beta0 + nbad
        pvalue[counter] = alpha / (alpha + beta)
        counter += 1

    while len(remove_set):
        remove_set = []
        # calculate the weighted sum of updates

        N = 0
        final_update = torch.zeros(flattened_weights[0].shape)
        counter = 0
        for delta_vector in flattened_weights:
            tmp = self.AFA_index_finder(
                delta_vector, retrive_flattened_weights[counter:]
            )
            print(tmp)
            if tmp != -1:
                index_value = tmp + counter
                N += pvalue[index_value]
                final_update += pvalue[index_value] * delta_vector
            counter = counter + 1

        final_update = final_update / N

        cos_sims = []

        for delta_vector in flattened_weights:
            cos_sim = (
                torch.dot(delta_vector.squeeze(), final_update.squeeze())
                / (torch.norm(final_update.squeeze()) + 1e-9)
                / (torch.norm(delta_vector.squeeze()) + 1e-9)
            )
            cos_sims = (
                cos_sim.unsqueeze(0)
                if not len(cos_sims)
                else torch.cat((cos_sims, cos_sim.unsqueeze(0)))
            )

        model_mean = torch.mean(cos_sims, dim=0).squeeze()
        model_median = torch.median(cos_sims, dim=0)[0].squeeze()
        model_std = torch.std(cos_sims, dim=0).squeeze()

        flattened_weights_copy = copy.deepcopy(flattened_weights)

        if model_mean < model_median:
            counter = 0
            for delta_vector in flattened_weights:
                if cos_sims[counter] < (model_median - epsilon * model_std):
                    remove_set.append(1)
                    remove_id = (
                        self.AFA_index_finder(
                            delta_vector, retrive_flattened_weights[counter:]
                        )
                        + counter
                    )
                    delete_id = self.AFA_index_finder(
                        delta_vector, flattened_weights_copy
                    )
                    temp_tensor1 = flattened_weights_copy[0:delete_id]
                    temp_tensor2 = flattened_weights_copy[delete_id + 1 :]
                    flattened_weights_copy = torch.cat(
                        (temp_tensor1, temp_tensor2), dim=0
                    )
                    bad_set.append(remove_id)
                    print(
                        counter,
                        remove_id,
                        delete_id,
                        len(retrive_flattened_weights),
                    )
                    print(torch.sum(torch.abs(delta_vector)))
                counter += 1
        else:
            counter = 0
            for delta_vector in flattened_weights:  #  we for loop this
                if cos_sims[counter] > (model_median + epsilon * model_std):
                    remove_set.append(1)
                    remove_id = (
                        self.AFA_index_finder(
                            delta_vector, retrive_flattened_weights[counter:]
                        )
                        + counter
                    )
                    delete_id = self.AFA_index_finder(
                        delta_vector, flattened_weights_copy
                    )
                    temp_tensor1 = flattened_weights_copy[0:delete_id]
                    temp_tensor2 = flattened_weights_copy[delete_id + 1 :]
                    flattened_weights_copy = torch.cat(
                        (temp_tensor1, temp_tensor2), dim=0
                    )  # but we changes it in the loop, maybe we should get a copy
                    bad_set.append(remove_id)

                counter += 1
        epsilon += delta_ep
        flattened_weights = copy.deepcopy(flattened_weights_copy)

    N = 0
    final_update = torch.zeros(flattened_weights[0].shape)
    counter = 0
    for delta_vector in flattened_weights:
        tmp = self.AFA_index_finder(delta_vector, retrive_flattened_weights[counter:])
        # print(tmp)
        if tmp != -1:
            index_value = tmp + counter
            N += pvalue[index_value]
            final_update += pvalue[index_value] * delta_vector
        counter = counter + 1

    final_update = final_update / N

    # update good_hist and bad_hist according to bad_set
    good_set = copy.deepcopy(clients_id)
    # print(bad_set)
    for rm_id in bad_set:
        # print(clients_id[29])
        self.bad_hist[clients_id[rm_id] - 1] += 1
        good_set.remove(clients_id[rm_id])
    for gd_id in good_set:
        self.good_hist[gd_id - 1] += 1

    for weight in weights_attacked:  # should iterate only once
        trimmed_mean_update = OrderedDict()
        for name in weight.keys():
            trimmed_mean_update[name] = final_update[
                start_index : start_index + len(weight[name].view(-1))
            ].reshape(weight[name].shape)
        start_index = start_index + len(weight[name].view(-1))
    return trimmed_mean_update


registered_aggregations = {
    "Median": median,
    "Bulyan": bulyan,
    "Krum": krum,
    "Afa": afa,
    "Trimmed-mean": trimmed_mean,
}
