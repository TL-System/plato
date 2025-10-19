"""
Aggregation strategies supporting the detector example.

These implementations mirror the original secure aggregation routines but expose
them through Plato's strategy interface so they can be composed with the server.
"""

import copy
import logging
import os
import pickle
from collections import OrderedDict
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import norm

from plato.config import Config
from plato.servers.strategies.base import AggregationStrategy, ServerContext


def flatten_weights(weights):
    """
    Flatten model weights into a 2D tensor where each row corresponds to a client.
    """
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


def reconstruct_weight(flattened_weight, reference):
    """Reconstruct model weights from a flattened tensor."""
    start_index = 0
    reconstructed = OrderedDict()
    for name, weight_value in reference.items():
        tensor = flattened_weight[
            start_index : start_index + len(weight_value.view(-1))
        ].reshape(weight_value.shape)
        reconstructed[name] = tensor
        start_index = start_index + len(weight_value.view(-1))
    return reconstructed


def configured_attacker_count():
    """Return the number of attackers defined in the configuration."""
    attacker_ids = getattr(Config().clients, "attacker_ids", "")

    if isinstance(attacker_ids, str):
        entries = [item.strip() for item in attacker_ids.split(",") if item.strip()]
        return len(entries)

    try:
        return len(attacker_ids)
    except TypeError:
        return 0


def median(updates, baseline_weights, weights_attacked):
    """Aggregate weight updates from the clients using coordinate-wise median."""
    if len(weights_attacked) == 0:
        logging.info("Median aggregation received no client updates.")
        return OrderedDict(baseline_weights)

    flattened_weights = flatten_weights(weights_attacked)
    median_weight = torch.median(flattened_weights, dim=0)[0]

    median_update = reconstruct_weight(median_weight, weights_attacked[0])

    logging.info("Finished Median server aggregation.")
    return median_update


def bulyan(updates, baseline_weights, weights_attacked):
    """Aggregate weight updates from the clients using Bulyan."""
    if len(weights_attacked) == 0:
        logging.info("Bulyan aggregation received no client updates.")
        return OrderedDict(baseline_weights)

    total_clients = getattr(Config().clients, "total_clients", len(weights_attacked))
    num_attackers = configured_attacker_count()

    remaining_weights = flatten_weights(weights_attacked)
    bulyan_cluster = []

    # Search for Bulyan cluster based on distances
    while len(remaining_weights) > 0 and (
        len(bulyan_cluster) < (total_clients - 2 * num_attackers)
    ):
        if len(remaining_weights) - 2 - num_attackers <= 0:
            break

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
        indices = torch.argsort(scores)
        if len(indices) == 0:
            break

        # Add candidates into Bulyan cluster
        bulyan_cluster = (
            remaining_weights[indices[0]][None, :]
            if not len(bulyan_cluster)
            else torch.cat((bulyan_cluster, remaining_weights[indices[0]][None, :]), 0)
        )

        # Remove candidate from remaining weights
        remaining_weights = torch.cat(
            (
                remaining_weights[: indices[0]],
                remaining_weights[indices[0] + 1 :],
            ),
            0,
        )

        if remaining_weights.shape[0] <= 2 * num_attackers + 2:
            break

    if not len(bulyan_cluster):
        logging.info("Bulyan cluster is empty, falling back to median aggregation.")
        return median(updates, baseline_weights, weights_attacked)

    # Perform sorting
    n, d = bulyan_cluster.shape
    trimmed = max(n - 2 * num_attackers, 1)
    median_weights = torch.median(bulyan_cluster, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(bulyan_cluster - median_weights), dim=0)
    sorted_weights = bulyan_cluster[sort_idx, torch.arange(d)[None, :]]

    # Average over sorted Bulyan cluster
    mean_weights = torch.mean(sorted_weights[:trimmed], dim=0)

    bulyan_update = reconstruct_weight(mean_weights, weights_attacked[0])

    logging.info("Finished Bulyan server aggregation.")
    return bulyan_update


def krum(updates, baseline_weights, weights_attacked):
    """Aggregate weight updates from the clients using Krum."""
    if len(weights_attacked) == 0:
        logging.info("Krum aggregation received no client updates.")
        return OrderedDict(baseline_weights)

    remaining_weights = flatten_weights(weights_attacked)

    if len(remaining_weights.shape) == 1:
        selected_weight = remaining_weights
    else:
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
        selected_weight = remaining_weights[sorted_scores[0]]

    krum_update = reconstruct_weight(selected_weight, weights_attacked[0])

    logging.info("Finished Krum server aggregation.")
    return krum_update


def multi_krum(updates, baseline_weights, weights_attacked):
    """Aggregate weight updates from the clients using Multi-Krum."""
    if len(weights_attacked) == 0:
        logging.info("Multi-Krum aggregation received no client updates.")
        return OrderedDict(baseline_weights)

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
        if len(indices) == 0:
            break

        candidates = (
            remaining_weights[indices[0]][None, :]
            if not len(candidates)
            else torch.cat((candidates, remaining_weights[indices[0]][None, :]), 0)
        )

        # Remove candidate from remaining weights
        remaining_weights = torch.cat(
            (
                remaining_weights[: indices[0]],
                remaining_weights[indices[0] + 1 :],
            ),
            0,
        )

    if not len(candidates):
        logging.info("No candidates selected for Multi-Krum; falling back to Krum.")
        return krum(updates, baseline_weights, weights_attacked)

    mean_weights = torch.mean(candidates, dim=0)
    mkrum_update = reconstruct_weight(mean_weights, weights_attacked[0])

    logging.info("Finished Multi-Krum server aggregation.")
    return mkrum_update


def trimmed_mean(updates, baseline_weights, weights_attacked):
    """Aggregate weight updates from the clients using trimmed mean."""
    if len(weights_attacked) == 0:
        logging.info("Trimmed-mean aggregation received no client updates.")
        return OrderedDict(baseline_weights)

    flattened_weights = flatten_weights(weights_attacked)
    num_attackers = configured_attacker_count()

    n, d = flattened_weights.shape
    trimmed = max(n - 2 * num_attackers, 1)
    median_weights = torch.median(flattened_weights, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(flattened_weights - median_weights), dim=0)
    sorted_weights = flattened_weights[sort_idx, torch.arange(d)[None, :]]

    mean_weights = torch.mean(sorted_weights[:trimmed], dim=0)

    trimmed_mean_update = reconstruct_weight(mean_weights, weights_attacked[0])

    logging.info("Finished Trimmed-mean server aggregation.")
    return trimmed_mean_update


def afa_index_finder(target_weight, all_weights):
    for counter, curr_weight in enumerate(all_weights):
        if target_weight.equal(curr_weight):
            return counter
    return -1


def afa(updates, baseline_weights, weights_attacked):
    """Aggregate weight updates from the clients using AFA."""
    if len(weights_attacked) == 0:
        logging.info("AFA aggregation received no client updates.")
        return OrderedDict(baseline_weights)

    flattened_weights = flatten_weights(weights_attacked)
    if isinstance(flattened_weights, list) or flattened_weights.numel() == 0:
        return OrderedDict(baseline_weights)

    clients_id = [update.client_id for update in updates]
    retrive_flattened_weights = flattened_weights.clone()

    bad_set = []
    remove_set = [1]
    pvalue: Dict[int, float] = {}
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
        good_hist = np.zeros(
            getattr(Config().clients, "total_clients", len(clients_id))
        )
        bad_hist = np.zeros(getattr(Config().clients, "total_clients", len(clients_id)))
        alpha = 3
        beta = 3

    for counter, client in enumerate(clients_id):
        ngood = good_hist[client - 1]
        nbad = bad_hist[client - 1]
        alpha = alpha + ngood
        beta = beta + nbad
        pvalue[counter] = alpha / (alpha + beta)

    final_update = torch.mean(flattened_weights, dim=0)

    # Search for bad actors
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

    # Update histories
    good_set = copy.deepcopy(clients_id)

    for rm_id in bad_set:
        bad_hist[clients_id[rm_id] - 1] += 1
        if clients_id[rm_id] in good_set:
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
            p_sum += pvalue.get(index_value, 0.0)
            final_update += pvalue.get(index_value, 0.0) * weight

    if p_sum != 0:
        final_update = final_update / p_sum
    else:
        final_update = torch.mean(flattened_weights, dim=0)

    afa_update = reconstruct_weight(final_update, weights_attacked[0])

    logging.info("Finished AFA server aggregation.")
    return afa_update


def fl_trust(updates, baseline, weights_attacked):
    """Aggregate weight updates from the clients using FL-Trust."""
    if len(weights_attacked) == 0:
        logging.info("FL-Trust aggregation received no client updates.")
        return OrderedDict(baseline)

    flattened_weights = flatten_weights(weights_attacked)
    num_clients, _ = flattened_weights.shape

    model_re = torch.mean(flattened_weights, dim=0).squeeze()
    cos_sims = []
    candidates = []

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

    cos_sims = torch.maximum(cos_sims, torch.tensor(0))
    normalized_weights = cos_sims / (torch.sum(cos_sims) + 1e-9)

    for i in range(num_clients):
        candidate = (
            flattened_weights[i]
            * normalized_weights[i]
            / (torch.norm(flattened_weights[i] + 1e-9))
            * torch.norm(model_re)
        )
        candidates = (
            candidate.unsqueeze(0)
            if not len(candidates)
            else torch.cat((candidates, candidate.unsqueeze(0)))
        )

    mean_weights = torch.sum(candidates, dim=0)
    avg_update = reconstruct_weight(mean_weights, weights_attacked[0])

    logging.info("Finished FL-Trust server aggregation.")
    return avg_update


class WeightsOnlyAggregationStrategy(AggregationStrategy):
    """Base class for aggregation strategies that work directly on model weights."""

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        raise NotImplementedError(
            "This aggregation strategy aggregates weights directly."
        )


class MedianAggregationStrategy(WeightsOnlyAggregationStrategy):
    async def aggregate_weights(
        self,
        updates: List[SimpleNamespace],
        baseline_weights: Dict,
        weights_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        return median(updates, baseline_weights, weights_received)


class BulyanAggregationStrategy(WeightsOnlyAggregationStrategy):
    async def aggregate_weights(
        self,
        updates: List[SimpleNamespace],
        baseline_weights: Dict,
        weights_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        return bulyan(updates, baseline_weights, weights_received)


class KrumAggregationStrategy(WeightsOnlyAggregationStrategy):
    async def aggregate_weights(
        self,
        updates: List[SimpleNamespace],
        baseline_weights: Dict,
        weights_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        return krum(updates, baseline_weights, weights_received)


class MultiKrumAggregationStrategy(WeightsOnlyAggregationStrategy):
    async def aggregate_weights(
        self,
        updates: List[SimpleNamespace],
        baseline_weights: Dict,
        weights_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        return multi_krum(updates, baseline_weights, weights_received)


class TrimmedMeanAggregationStrategy(WeightsOnlyAggregationStrategy):
    async def aggregate_weights(
        self,
        updates: List[SimpleNamespace],
        baseline_weights: Dict,
        weights_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        return trimmed_mean(updates, baseline_weights, weights_received)


class AfaAggregationStrategy(WeightsOnlyAggregationStrategy):
    async def aggregate_weights(
        self,
        updates: List[SimpleNamespace],
        baseline_weights: Dict,
        weights_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        return afa(updates, baseline_weights, weights_received)


class FLTrustAggregationStrategy(WeightsOnlyAggregationStrategy):
    async def aggregate_weights(
        self,
        updates: List[SimpleNamespace],
        baseline_weights: Dict,
        weights_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        return fl_trust(updates, baseline_weights, weights_received)


__all__ = [
    "MedianAggregationStrategy",
    "BulyanAggregationStrategy",
    "KrumAggregationStrategy",
    "MultiKrumAggregationStrategy",
    "TrimmedMeanAggregationStrategy",
    "AfaAggregationStrategy",
    "FLTrustAggregationStrategy",
]
