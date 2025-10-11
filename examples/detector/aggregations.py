"""
The registry that contains all available secure aggregation in federated learning.

Having a registry of all available classes is convenient for retrieving an instance based
on a configuration at run-time.
"""

import copy
import logging
import os
import pickle
from collections import OrderedDict
from typing import Mapping

import numpy as np
import torch
from scipy.stats import norm

from plato.config import Config


def get():
    """Get a secure aggregation method based on the configuration file."""
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

    median_weight = torch.median(flattened_weights, dim=0)[0]

    # Update global model
    start_index = 0
    median_update = OrderedDict()
    for name, weight_value in weights_attacked[0].items():
        median_update[name] = median_weight[
            start_index : start_index + len(weight_value.view(-1))
        ].reshape(weight_value.shape)
        start_index = start_index + len(weight_value.view(-1))

    logging.info(f"Finished Median server aggregation.")

    return median_update


def bulyan(updates, baseline_weights, weights_attacked):
    """Aggregate weight updates from the clients using bulyan."""

    total_clients = Config().clients.total_clients
    num_attackers = len(Config().clients.attacker_ids)

    remaining_weights = flatten_weights(weights_attacked)
    bulyan_cluster = []

    # Search for bulyan cluster based on distances
    while (len(bulyan_cluster) < (total_clients - 2 * num_attackers)) and (
        len(bulyan_cluster) < (total_clients - 2 - num_attackers)
    ):
        distances = []
        for weight in remaining_weights:
            distance = torch.norm((remaining_weights - weight), dim=1) ** 2
            distances = (
                distance[None, :]
                if not len(distances)
                else torch.cat((distances, distance[None, :]), 0)
            )

        distances = torch.sort(distances, dim=1)[0]

        scores = torch.sum(
            distances[:, : len(remaining_weights) - 2 - num_attackers], dim=1
        )
        indices = torch.argsort(scores)[: len(remaining_weights) - 2 - num_attackers]

        # Add candidates into bulyan cluster
        bulyan_cluster = (
            remaining_weights[indices[0]][None, :]
            if not len(bulyan_cluster)
            else torch.cat((bulyan_cluster, remaining_weights[indices[0]][None, :]), 0)
        )

        # Remove candidates from remainings
        remaining_weights = torch.cat(
            (
                remaining_weights[: indices[0]],
                remaining_weights[indices[0] + 1 :],
            ),
            0,
        )

    # Perform sorting
    n, d = bulyan_cluster.shape
    median_weights = torch.median(bulyan_cluster, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(bulyan_cluster - median_weights), dim=0)
    sorted_weights = bulyan_cluster[sort_idx, torch.arange(d)[None, :]]

    # Average over sorted bulyan cluster
    mean_weights = torch.mean(sorted_weights[: n - 2 * num_attackers], dim=0)

    # Update global model
    start_index = 0
    bulyan_update = OrderedDict()
    for name, weight_value in weights_attacked[0].items():
        bulyan_update[name] = mean_weights[
            start_index : start_index + len(weight_value.view(-1))
        ].reshape(weight_value.shape)
        start_index = start_index + len(weight_value.view(-1))

    logging.info(f"Finished Bulyan server aggregation.")
    return bulyan_update


def krum(updates, baseline_weights, weights_attacked):
    """Aggregate weight updates from the clients using krum."""

    remaining_weights = flatten_weights(weights_attacked)

    num_attackers_selected = 2

    distances = []
    for weight in remaining_weights:
        distance = torch.norm((remaining_weights - weight), dim=1) ** 2
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

    # Pick the top-1 candidate
    candidates_weights = remaining_weights[sorted_scores[0]][None, :]

    logging.info(f"Finished krum server aggregation.")
    return candidates_weights


def multi_krum(updates, baseline_weights, weights_attacked):
    """Aggregate weight updates from the clients using multi-krum."""
    remaining_weights = flatten_weights(weights_attacked)

    num_attackers_selected = 2
    candidates = []

    # Search for candidates based on distance
    while len(remaining_weights) > 2 * num_attackers_selected + 2:
        distances = []
        for weight in remaining_weights:
            distance = torch.norm((remaining_weights - weight), dim=1) ** 2
            distances = (
                distance[None, :]
                if not len(distances)
                else torch.cat((distances, distance[None, :]), 0)
            )

        distances = torch.sort(distances, dim=1)[0]

        scores = torch.sum(
            distances[:, : len(remaining_weights) - 2 - num_attackers_selected],
            dim=1,
        )
        indices = torch.argsort(scores)
        candidates = (
            remaining_weights[indices[0]][None, :]
            if not len(candidates)
            else torch.cat((candidates, remaining_weights[indices[0]][None, :]), 0)
        )

        # Remove candidates from remainings
        remaining_weights = torch.cat(
            (
                remaining_weights[: indices[0]],
                remaining_weights[indices[0] + 1 :],
            ),
            0,
        )

    mean_weights = torch.mean(candidates, dim=0)

    # Update global model
    start_index = 0
    mkrum_update = OrderedDict()
    for name, weight_value in weights_attacked[0].items():
        mkrum_update[name] = mean_weights[
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
    median_weights = torch.median(flattened_weights, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(flattened_weights - median_weights), dim=0)
    sorted_weights = flattened_weights[sort_idx, torch.arange(d)[None, :]]

    mean_weights = torch.mean(sorted_weights[: n - 2 * num_attackers], dim=0)

    start_index = 0
    trimmed_mean_update = OrderedDict()
    for name, weight_value in weights_attacked[0].items():
        trimmed_mean_update[name] = mean_weights[
            start_index : start_index + len(weight_value.view(-1))
        ].reshape(weight_value.shape)
        start_index = start_index + len(weight_value.view(-1))

    logging.info(f"Finished Trimmed mean server aggregation.")

    return trimmed_mean_update


def afa_index_finder(target_weight, all_weights):
    for counter, curr_weight in enumerate(all_weights):
        if target_weight.equal(curr_weight):
            return counter
    return -1


def afa(updates, baseline_weights, weights_attacked):
    """Aggregate weight updates from the clients using afa."""

    flattened_weights = flatten_weights(weights_attacked)
    clients_id = [update.client_id for update in updates]

    retrive_flattened_weights = flattened_weights.clone()

    bad_set = []
    remove_set = [1]
    pvalue = {}
    epsilon = 2
    delta_ep = 0.5

    # Load from the history or create new ones
    file_path = "./parameters.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            good_hist = pickle.load(file)
            bad_hist = pickle.load(file)
            alpha = pickle.load(file)
            beta = pickle.load(file)
    else:
        good_hist = np.zeros(Config().clients.total_clients)
        bad_hist = np.zeros(Config().clients.total_clients)
        alpha = 3
        beta = 3

    for counter, client in enumerate(clients_id):
        ngood = good_hist[client - 1]
        nbad = bad_hist[client - 1]
        alpha = alpha + ngood
        beta = beta + nbad
        pvalue[counter] = alpha / (alpha + beta)

    # Search for bad guys
    while len(remove_set):
        remove_set = []

        cos_sims = []
        for weight in flattened_weights:
            cos_sim = (
                torch.dot(weight.squeeze(), final_update.squeeze())
                / (torch.norm(final_update.squeeze()) + 1e-9)
                / (torch.norm(weight.squeeze()) + 1e-9)
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
            for counter, weight in enumerate(flattened_weights):
                if cos_sims[counter] < (model_median - epsilon * model_std):
                    remove_set.append(1)
                    remove_id = (
                        afa_index_finder(weight, retrive_flattened_weights[counter:])
                        + counter
                    )
                    delete_id = afa_index_finder(weight, flattened_weights_copy)
                    temp_tensor1 = flattened_weights_copy[0:delete_id]
                    temp_tensor2 = flattened_weights_copy[delete_id + 1 :]
                    flattened_weights_copy = torch.cat(
                        (temp_tensor1, temp_tensor2), dim=0
                    )
                    bad_set.append(remove_id)

        else:
            for counter, weight in enumerate(flattened_weights):
                if cos_sims[counter] > (model_median + epsilon * model_std):
                    remove_set.append(1)
                    remove_id = (
                        afa_index_finder(weight, retrive_flattened_weights[counter:])
                        + counter
                    )
                    delete_id = afa_index_finder(weight, flattened_weights_copy)
                    temp_tensor1 = flattened_weights_copy[0:delete_id]
                    temp_tensor2 = flattened_weights_copy[delete_id + 1 :]
                    flattened_weights_copy = torch.cat(
                        (temp_tensor1, temp_tensor2), dim=0
                    )
                    bad_set.append(remove_id)

        epsilon += delta_ep
        flattened_weights = copy.deepcopy(flattened_weights_copy)

    # Update good_hist and bad_hist according to bad_set
    good_set = copy.deepcopy(clients_id)

    # Update history
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

    # Perform aggregation
    p_sum = 0
    final_update = torch.zeros(flattened_weights[0].shape)

    for counter, weight in enumerate(flattened_weights):
        tmp = afa_index_finder(weight, retrive_flattened_weights[counter:])
        if tmp != -1:
            index_value = tmp + counter
            p_sum += pvalue[index_value]
            final_update += pvalue[index_value] * weight

    final_update = final_update / p_sum

    # Update globel weights
    start_index = 0
    afa_update = OrderedDict()
    for name, weight_value in weights_attacked[0].items():
        afa_update[name] = final_update[
            start_index : start_index + len(weight_value.view(-1))
        ].reshape(weight_value.shape)
        start_index = start_index + len(weight_value.view(-1))

    logging.info(f"Finished AFA server aggregation.")
    return afa_update


def fl_trust(updates, baseline, weights_attacked):
    """Aggregate weight updates from the clients using fltrust."""
    flattened_weights = flatten_weights(weights_attacked)
    num_clients, d = flattened_weights.shape

    model_re = torch.mean(flattened_weights, dim=0).squeeze()
    cos_sims = []
    candidates = []

    # compute cos similarity
    for weight in flattened_weights:
        cos_sim = (
            torch.dot(weight.squeeze(), model_re)
            / (torch.norm(model_re) + 1e-9)
            / (torch.norm(weight.squeeze()) + 1e-9)
        )
        cos_sims = (
            cos_sim.unsqueeze(0)
            if not len(cos_sims)
            else torch.cat((cos_sims, cos_sim.unsqueeze(0)))
        )

    # ReLU
    cos_sims = torch.maximum(cos_sims, torch.tensor(0))
    normalized_weights = cos_sims / (torch.sum(cos_sims) + 1e-9)
    for i in range(num_clients):
        candidate = (
            flattened_weights[i]
            * normalized_weights[i]
            / torch.norm(flattened_weights[i] + 1e-9)
            * torch.norm(model_re)
        )
        candidates = (
            candidate.unsqueeze(0)
            if not len(candidates)
            else torch.cat((candidates, candidate.unsqueeze(0)))
        )

    mean_weights = torch.sum(candidates, dim=0)

    # Update global model
    start_index = 0
    avg_update = OrderedDict()
    for name, weight_value in weights_attacked[0].items():
        avg_update[name] = mean_weights[
            start_index : start_index + len(weight_value.view(-1))
        ].reshape(weight_value.shape)
        start_index = start_index + len(weight_value.view(-1))

    logging.info(f"Finished FL-trust server aggregation.")
    return avg_update


registered_aggregations = {
    "Median": median,
    "Bulyan": bulyan,
    "Krum": krum,
    "Multi-krum": multi_krum,
    "Afa": afa,
    "Trimmed-mean": trimmed_mean,
    "Fl-trust": fl_trust,
}
