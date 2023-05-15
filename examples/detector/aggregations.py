import torch
import logging
from plato.config import Config
from scipy.stats import norm
from collections import OrderedDict
import numpy as np
import copy
from typing import Mapping
import os
import pickle


def get():

    aggregation_type = (
        Config().server.secure_aggregation_type
        if hasattr(Config().server, "secure_aggregation_type")
        else None
    )

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
    # Update global model
    start_index = 0
    median_update = OrderedDict()
    for name, weight_value in weights_attacked[0].items():
        median_update[name] = median_delta_vector[
            start_index : start_index + len(weight_value.view(-1))
        ].reshape(weight_value.shape)
        start_index = start_index + len(weight_value.view(-1))

    logging.info(f"Finished Median server aggregation.")

    return median_update


def bulyan(updates, baseline_weights, weights_attacked):
    """Aggregate weight updates from the clients using bulyan."""

    total_clients = Config().clients.total_clients
    num_attackers = len(Config().clients.attacker_ids)  # ?

    remaining_deltas_vector = flatten_weights(weights_attacked)
    bulyan_cluster = []

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
    # perform sorting
    n, d = bulyan_cluster.shape
    param_median = torch.median(bulyan_cluster, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(bulyan_cluster - param_median), dim=0)
    sorted_deltas_vector = bulyan_cluster[sort_idx, torch.arange(d)[None, :]]

    # average over sorted bulyan cluster
    mean_delta_vector = torch.mean(sorted_deltas_vector[: n - 2 * num_attackers], dim=0)
    # Update global model
    start_index = 0
    bulyan_update = OrderedDict()
    for name, weight_value in weights_attacked[0].items():
        bulyan_update[name] = mean_delta_vector[
            start_index : start_index + len(weight_value.view(-1))
        ].reshape(weight_value.shape)
        start_index = start_index + len(weight_value.view(-1))

    logging.info(f"Finished Bulyan server aggregation.")
    return bulyan_update


def krum(updates, baseline_weights, weights_attacked):
    """Aggregate weight updates from the clients using krum."""

    remaining_weights = flatten_weights(weights_attacked)

    num_attackers_selected = 2  # ?

    distances = []
    for deltas_vector in remaining_weights:
        distance = torch.norm((remaining_weights - deltas_vector), dim=1) ** 2
        distances = (
            distance[None, :]
            if not len(distances)
            else torch.cat((distances, distance[None, :]), 0)
        )

    distances = torch.sort(distances, dim=1)[0]
    scores = torch.sum(
        distances[:, : len(remaining_weights) - 2 - num_attackers_selected], dim=1
    )
    sorted_scores = torch.argsort(scores)
    # put candidates' weights in to candidates (top1)
    candidates_weights = remaining_weights[sorted_scores[0]][None, :]

    logging.info(f"Finished krum server aggregation.")
    return candidates_weights


def multi_krum(updates, baseline_weights, weights_attacked):

    """Aggregate weight updates from the clients using multi-krum."""
    remaining_deltas_vector = flatten_weights(weights_attacked)

    num_attackers_selected = 2
    candidates = []

    while len(remaining_deltas_vector) > 2 * num_attackers_selected + 2:
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
            distances[:, : len(remaining_deltas_vector) - 2 - num_attackers_selected],
            dim=1,
        )
        indices = torch.argsort(scores)
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

    # Update global model
    start_index = 0
    mkrum_update = OrderedDict()
    for name, weight_value in weights_attacked[0].items():
        mkrum_update[name] = mean_delta_vector[
            start_index : start_index + len(weight_value.view(-1))
        ].reshape(weight_value.shape)
        start_index = start_index + len(weight_value.view(-1))
        
    logging.info(f"Finished multi-krum server aggregation.")
    return mkrum_update

def trimmed_mean(updates, baseline_weights, weights_attacked):
    """Aggregate weight updates from the clients using trimmed-mean."""
    flattened_weights = flatten_weights(weights_attacked)
    num_attackers = 0  # ?

    n, d = flattened_weights.shape
    param_median = torch.median(flattened_weights, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(flattened_weights - param_median), dim=0)
    sorted_deltas_vector = flattened_weights[sort_idx, torch.arange(d)[None, :]]

    mean_delta_vector = torch.mean(sorted_deltas_vector[: n - 2 * num_attackers], dim=0)

    start_index = 0
    median_update = OrderedDict()
    for name, weight_value in weights_attacked[0].items():
        median_update[name] = mean_delta_vector[
            median_update[name] = mean_delta_vector[
        ].reshape(weight_value.shape)
        start_index = start_index + len(weight_value.view(-1))

    logging.info(f"Finished Trimmed mean server aggregation.")

    return median_update


        
    counter = 0
    for curr_delta in retrive_all_delta:
        if input_delta.equal(curr_delta):
            return counter
        counter += 1
    return -1


def afa(updates, baseline_weights, weights_attacked):
    """Aggregate weight updates from the clients using afa."""

    flattened_weights = flatten_weights(weights_attacked)
    # logging.info(f"updates: %s",updates)

    clients_id = [update.client_id for update in updates]

    retrive_flattened_weights = flattened_weights.clone()

    bad_set = []
    remove_set = [1]
    pvalue = {}

    epsilon = 2
    delta_ep = 0.5

    counter = 0
    file_path = "./parameters.pkl"
    if os.path.exists(file_path):
        # Open the file in binary read mode
        with open(file_path, "rb") as file:
            with open(file_path, 'rb') as file:
            good_hist = pickle.load(file)
            bad_hist = pickle.load(file)
            alpha = pickle.load(file)
            beta = pickle.load(file)
    else:
        good_hist = np.zeros(Config().clients.total_clients)
        bad_hist = np.zeros(Config().clients.total_clients)
        alpha = 3
        beta = 3

    for client in clients_id:
        ngood = good_hist[client - 1]
        nbad = bad_hist[client - 1]
        alpha = alpha + ngood
        beta = beta + nbad
        pvalue[counter] = alpha / (alpha + beta)
        counter += 1

    while len(remove_set):
        remove_set = []
        # calculate the weighted sum of updates
            flattened_weights = copy.deepcopy(flattened_weights_copy)

        N = 0
        final_update = torch.zeros(flattened_weights[0].shape)
        counter = 0
            tmp = AFA_index_finder(delta_vector, retrive_flattened_weights[counter:])
            )
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
                        AFA_index_finder(
                            delta_vector, retrive_flattened_weights[counter:]
                        )
                        + counter
                    )
                    delete_id = AFA_index_finder(delta_vector, flattened_weights_copy)
                    temp_tensor1 = flattened_weights_copy[0:delete_id]
                    temp_tensor2 = flattened_weights_copy[delete_id + 1 :]
                    flattened_weights_copy = torch.cat(
                        (temp_tensor1, temp_tensor2), dim=0
                    )
                    bad_set.append(remove_id)
                counter += 1
        else:
            counter = 0
            for delta_vector in flattened_weights:  #  we for loop this
                if cos_sims[counter] > (model_median + epsilon * model_std):
                    remove_set.append(1)
                    remove_id = (
                        AFA_index_finder(
                            delta_vector, retrive_flattened_weights[counter:]
                        )
                        + counter
                    )
                    delete_id = AFA_index_finder(delta_vector, flattened_weights_copy)
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
        tmp = AFA_index_finder(delta_vector, retrive_flattened_weights[counter:])
        if tmp != -1:
            index_value = tmp + counter
            N += pvalue[index_value]
            final_update += pvalue[index_value] * delta_vector
        counter = counter + 1

    final_update = final_update / N

    # update good_hist and bad_hist according to bad_set
    good_set = copy.deepcopy(clients_id)

    for rm_id in bad_set:
        bad_hist[clients_id[rm_id] - 1] += 1
        good_set.remove(clients_id[rm_id])
    for gd_id in good_set:
        good_hist[gd_id - 1] += 1
    with open(file_path, "wb") as file:
        pickle.dump(good_hist, file)
        pickle.dump(bad_hist, file)
        pickle.dump(alpha, file)
        pickle.dump(beta, file)

    start_index = 0
    afa_update = OrderedDict()
    for name, weight_value in weights_attacked[0].items():
        afa_update[name] = final_update[
            afa_update[name] = final_update[
        ].reshape(weight_value.shape)
        start_index = start_index + len(weight_value.view(-1))
    logging.info(f"Finished AKA server aggregation.")
    return afa_update


registered_aggregations = {
    "Median": median,
    "Bulyan": bulyan,
    "Krum": krum,
    "Afa": afa,
    "Trimmed-mean": trimmed_mean,
}
registered_aggregations = {"Median":median, "Bulyan": bulyan,"Krum":krum,"Multi-krum":multi_krum,"Afa": afa, "Trimmed-mean": trimmed_mean}