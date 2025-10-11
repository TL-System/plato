"""
The registry that contains all available model poisoning attacks in federated learning.

Having a registry of all available classes is convenient for retrieving an instance based
on a configuration at run-time.
"""

import logging
import os
import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm

from plato.config import Config


def get():
    """Get an attack for malicious clients based on the configuration file."""
    attack_type = (
        Config().clients.attack_type
        if hasattr(Config().clients, "attack_type")
        else None
    )

    if attack_type is None:
        logging.info(f"No attack is applied.")
        return lambda a, x, b, c: x

    if attack_type in registered_attacks:
        registered_attack = registered_attacks[attack_type]
        logging.info(f"Clients perform {attack_type} attack.")
        return registered_attack

    raise ValueError(f"No such attack: {attack_type}")


def poisoning_performance_evaluation(clean_weights, poisoned_values, poisoned_weights):
    # calculate deviated norm of poisoned weights from clean weights
    # poisoned_values is a long tensor, and it's same for all clients
    logging.info(
        f"poisoning performance evaluation (torch.norm(poisoned_value)): %s",
        torch.norm(poisoned_values),
    )
    logging.info(f"norm of clean weights are: %s", torch.norm(clean_weights, dim=1))
    flatten_poisoned_weights = flatten_weights(poisoned_weights)
    logging.info(
        f"norm of poisoned weights: %s", torch.norm(flatten_poisoned_weights, dim=1)
    )


def perform_model_poisoning(weights_received, poison_value):
    # Poison the received weights based on calculated poison value.
    # The "poison_value" means modifications on original weights
    for index, weight_received in enumerate(weights_received):
        start_index = 0

        for name, weight in weight_received.items():
            if weights_received[index][name].dtype == torch.int64:
                weights_received[index][name] += (
                    poison_value[start_index : start_index + len(weight.view(-1))]
                    .reshape(weight.shape)
                    .long()
                )

            else:
                weights_received[index][name] += poison_value[
                    start_index : start_index + len(weight.view(-1))
                ].reshape(weight.shape)
            start_index += len(weight.view(-1))

    return weights_received


def flatten_weight(weight):
    flattened_weight = []
    for name in weight.keys():
        flattened_weight = (
            weight[name].view(-1)
            if not len(flattened_weight)
            else torch.cat((flattened_weight, weight[name].view(-1)))
        )
    return flattened_weight


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


def compute_sali_indicator():
    # Add importance pruning to the attack
    sali_map_vector = torch.load(Config().algorithm.map_path)

    sparsity = Config().algorithm.sparsity
    shrink_level = Config().algorithm.shrink_level
    inflation_level = Config().algorithm.inflation_level

    thresh = float(
        sali_map_vector.kthvalue(int(sparsity * sali_map_vector.shape[0]))[0]
    )

    sali_indicators_vector = torch.zeros(sali_map_vector.shape)
    sali_indicators_vector[sali_map_vector <= thresh] = shrink_level
    sali_indicators_vector[sali_map_vector > thresh] = inflation_level

    return sali_indicators_vector


def smoothing(keywords, value):
    total_clients = Config().clients.total_clients
    num_attackers = len(Config().clients.attacker_ids)
    clients_per_round = Config().clients.per_round

    malicious_expectation = (num_attackers / total_clients) * clients_per_round
    if num_attackers < malicious_expectation:
        momentum = Config().algorithm.high_momentum
    else:
        momentum = Config().algorithm.low_momentum
    # Smooth poison value
    file_path = "./" + keywords + "_model_updates_history.pt"
    if os.path.exists(file_path):
        last_model_re = torch.load(file_path)
        value = (1 - momentum) * value + momentum * last_model_re

    torch.save(value, file_path)
    return value


def gassian_attack(baseline_weights, weights_received, deltas_received, num_attackers):
    # calculate poison value based on Gassian distribution
    attacker_weights = flatten_weights(weights_received)
    baseline_weights = flatten_weight(baseline_weights)

    # set standard deviation and mean value for gassian noise
    std_dev = 1
    mean = 0
    poison_value = torch.randn_like(attacker_weights[0]) * std_dev + mean

    # Perform model poisoning
    weights_poisoned = perform_model_poisoning(weights_received, poison_value)
    logging.info(f"Finished Gassian model poisoning attack.")
    # poisoning_performance_evaluation(attacker_weights, poison_value, weights_poisoned)
    return weights_poisoned


def lambda_attack(baseline_weights, weights_received, deltas_received, num_attackers):
    attacker_weights = flatten_weights(weights_received)
    attacker_deltas = flatten_weights(deltas_received)
    baseline_weights = flatten_weight(baseline_weights)

    lamda = Config().clients.lambada_value
    direction = -1  # opposite direction;
    perturbation_vector = direction * torch.mean(attacker_deltas, dim=0)
    poison_value = lamda * perturbation_vector
    # deltas_poisoned = perform_model_poisoning(deltas_received, poison_value)

    # Perform model poisoning
    weights_poisoned = perform_model_poisoning(weights_received, poison_value)
    logging.info(f"Finished Lambda model poisoning attack.")
    # poisoning_performance_evaluation(attacker_weights, poison_value, weights_poisoned)

    return weights_poisoned


def lie_attack(baseline_weights, weights_received, deltas_received, num_attackers):
    """
    Attack name: Little is enough

    Reference:

    Baruch et al., "A little is enough: Circumventing defenses for distributed learning," in Proceedings of Advances in Neural Information Processing Systems (NeurIPS) 2019.

    https://proceedings.neurips.cc/paper_files/paper/2019/file/ec1c59141046cd1866bbbcdfb6ae31d4-Paper.pdf
    """
    if num_attackers == 1:
        # LIE attack does not apply to solo attacker
        return weights_received
    else:
        total_clients = Config().clients.per_round
        # num_attackers = len(Config().clients.attacker_ids)

        attacker_weights = flatten_weights(weights_received)
        attacker_deltas = flatten_weights(deltas_received)
        baseline_weights = flatten_weight(baseline_weights)

        # Calculate perturbation range
        s_value = total_clients / 2 + 1 - num_attackers
        possibility = (total_clients - s_value) / total_clients
        z_value = norm.cdf(possibility)

        deltas_avg = torch.mean(attacker_deltas, dim=0)
        weights_std = -1 * torch.std(attacker_deltas, dim=0)

        # Calculate poisoning model
        poison_value = z_value * weights_std
        # Perform model poisoning
        weights_poisoned = perform_model_poisoning(weights_received, poison_value)
        logging.info(f"Finished LIE model poisoning attack.")
        # poisoning_performance_evaluation(attacker_weights, poison_value, weights_poisoned)
        return weights_poisoned


def oblivion_lie_attack(weights_received, deltas_received, num_attackers):
    """Little is enough"""
    """     LIE importance dynamic momentum attack   """
    total_clients = Config().clients.per_round
    # num_attackers = len(Config().clients.attacker_ids)

    attacker_weights = flatten_weights(weights_received)
    attacker_deltas = flatten_weights(deltas_received)

    # Calculate perturbation range
    s_value = total_clients / 2 + 1 - num_attackers
    possibility = (total_clients - s_value) / total_clients
    z_value = norm.cdf(possibility)

    weights_avg = torch.mean(attacker_weights, dim=0)
    weights_std = torch.std(attacker_weights, dim=0)

    # Smooth benign model updates
    weights_avg = smoothing("benign", weights_avg)

    # Importance pruning
    sali_indicators_vector = compute_sali_indicator()
    weights_std = weights_std * sali_indicators_vector.cpu()

    poison_value = weights_avg + z_value * weights_std

    # Smooth poison value
    poison_value = smoothing("poisoned", poison_value)

    # Perform model poisoning
    weights_poisoned = perform_model_poisoning(weights_received, poison_value)
    logging.info(f"Finished LIE Oblivion model poisoning attack.")
    # poisoning_performance_evaluation(attacker_weights, poison_value, weights_poisoned)

    return weights_poisoned


def min_max_attack(
    baseline_weights,
    weights_received,
    deltas_received,
    num_attackers,
    dev_type="unit_vec",
):
    # attacker_weights = flatten_weights(weights_received)
    attacker_weights = flatten_weights(deltas_received)

    # weights_avg = torch.mean(attacker_weights, 0)
    weights_avg = torch.mean(attacker_weights, 0)

    # Generate perturbation vectors (Inverse unit vector by default)
    if dev_type == "unit_vec":
        # Inverse unit vector
        perturbation_vector = weights_avg / torch.norm(weights_avg)
    elif dev_type == "sign":
        # Inverse sign
        perturbation_vector = torch.sign(weights_avg)
    elif dev_type == "std":
        # Inverse standard deviation
        perturbation_vector = torch.std(attacker_weights, 0)

    # Calculate the maximum distance between any two benign updates (unpoisoned)
    max_distance = torch.tensor([0])
    for attacker_weight in attacker_weights:
        distance = torch.norm((attacker_weights - attacker_weight), dim=1) ** 2
        max_distance = torch.max(max_distance, torch.max(distance))
    # Search for lambda such that its maximum distance from any other gradient is bounded
    lambda_value = torch.Tensor([10000.0]).float()
    threshold = 1e-5
    lambda_step = lambda_value
    lambda_succ = 0

    while torch.abs(lambda_succ - lambda_value) > threshold:
        poison_value = weights_avg - lambda_value * perturbation_vector
        distance = torch.norm((attacker_weights - poison_value), dim=1) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            lambda_succ = lambda_value
            lambda_value = lambda_value + lambda_step / 2
        else:
            lambda_value = lambda_value - lambda_step / 2

        lambda_step = lambda_step / 2

    # poison_value = weights_avg - lambda_succ * perturbation_vector
    poison_value = -1 * lambda_succ * perturbation_vector
    # Perform model poisoning
    weights_poisoned = perform_model_poisoning(weights_received, poison_value)
    logging.info(f"Finished Min-Max model poisoning attack.")
    # poisoning_performance_evaluation(attacker_weights, poison_value, weights_poisoned)
    return weights_poisoned


def oblivion_min_max_attack(weights_received, dev_type="unit_vec"):
    """
    Attack: Min-Max

    Reference:

    Shejwalkar et al., “Manipulating the Byzantine: Optimizing model poisoning attacks and defenses for federated learning,” in Proceedings of 28th Annual Network and Distributed System Security Symposium (NDSS), 2021

    https://www.ndss-symposium.org/ndss-paper/manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/
    """

    attacker_weights = flatten_weights(weights_received)

    weights_avg = torch.mean(attacker_weights, 0)

    # Smooth benign model updates
    weights_avg = smoothing("benign", weights_avg)

    # Generate perturbation vectors (Inverse unit vector by default)
    if dev_type == "unit_vec":
        # Inverse unit vector
        perturbation_vector = weights_avg / torch.norm(weights_avg)
    elif dev_type == "sign":
        # Inverse sign
        perturbation_vector = torch.sign(weights_avg)
    elif dev_type == "std":
        # Inverse standard deviation
        perturbation_vector = torch.std(attacker_weights, 0)

    # Importance pruning
    sali_indicators_vector = compute_sali_indicator()
    perturbation_vector = perturbation_vector * sali_indicators_vector

    # Calculate the maximum distance between any two benign updates (unpoisoned)
    max_distance = torch.tensor([0])
    for attacker_weight in attacker_weights:
        distance = torch.norm((attacker_weights - attacker_weight), dim=1) ** 2
        max_distance = torch.max(max_distance, torch.max(distance))

    # Search for lambda such that its maximum distance from any other gradient is bounded
    lambda_value = torch.Tensor([50.0]).float()
    threshold = 1  # e-3 #1e-5
    lambda_step = lambda_value
    lambda_succ = 0

    while torch.abs(lambda_succ - lambda_value) > threshold:
        poison_value = weights_avg - lambda_value * perturbation_vector
        distance = torch.norm((attacker_weights - poison_value), dim=1) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            lambda_succ = lambda_value
            lambda_value = lambda_value + lambda_step / 2
        else:
            lambda_value = lambda_value - lambda_step / 2

        lambda_step = lambda_step / 2

    poison_value = weights_avg - lambda_succ * perturbation_vector

    # Smooth poison value
    poison_value = smoothing("poisoned", poison_value)

    # Perform model poisoning
    weights_poisoned = perform_model_poisoning(weights_received, poison_value)
    logging.info(f"Finished Min-Max model poisoning attack (with Oblivion).")
    return weights_poisoned


def min_sum_attack(
    baseline_weights,
    weights_received,
    deltas_received,
    num_attackers,
    dev_type="unit_vec",
):
    """
    Attack: Min-Sum

    Reference:

    Shejwalkar et al., “Manipulating the Byzantine: Optimizing model poisoning attacks and defenses for federated learning,” in Proceedings of 28th Annual Network and Distributed System Security Symposium (NDSS), 2021

    https://www.ndss-symposium.org/ndss-paper/manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/
    """

    attacker_weights = flatten_weights(weights_received)
    attacker_deltas = flatten_weights(deltas_received)

    # deltas_avg = torch.mean(attacker_deltas, 0)
    weights_avg = torch.mean(attacker_weights, 0)

    # Generate perturbation vectors (Inverse unit vector by default)
    if dev_type == "unit_vec":
        # Inverse unit vector
        perturbation_vector = weights_avg / torch.norm(weights_avg)
    elif dev_type == "sign":
        # Inverse sign
        perturbation_vector = torch.sign(weights_avg)
    elif dev_type == "std":
        # Inverse standard deviation
        perturbation_vector = torch.std(attacker_deltas, 0)

    # Calculate the minimal sum of squared distances of benign update from the other benign updates
    min_sum_distance = torch.tensor([50000000000])
    for attacker_weight in attacker_deltas:
        distance = torch.norm((attacker_deltas - attacker_weight), dim=1) ** 2
        min_sum_distance = torch.min(min_sum_distance, torch.sum(distance))

    # Search for lambda
    lambda_value = torch.Tensor([10000.0]).float()
    threshold = 1e-5
    lambda_step = lambda_value
    lambda_succ = 0

    while torch.abs(lambda_succ - lambda_value) > threshold:
        poison_value = weights_avg - lambda_value * perturbation_vector
        distance = torch.norm((attacker_deltas - poison_value), dim=1) ** 2
        score = torch.sum(distance)
        if score <= min_sum_distance:
            lambda_succ = lambda_value
            lambda_value = lambda_value + lambda_step / 2
        else:
            lambda_succ = lambda_value  # incase no succ

            lambda_value = lambda_value - lambda_step / 2

        lambda_step = lambda_step / 2
    poison_value = lambda_succ * perturbation_vector
    # perform model poisoning
    weights_poisoned = perform_model_poisoning(weights_received, poison_value)
    logging.info(f"Finished Min-Sum model poisoning attack.")
    # poisoning_performance_evaluation(attacker_weights, poison_value, weights_poisoned)
    return weights_poisoned


def oblivion_min_sum_attack(weights_received, dev_type="unit_vec"):
    """
    Attack: Min-Max

    Reference:

    Shejwalkar et al., “Manipulating the Byzantine: Optimizing model poisoning attacks and defenses for federated learning,” in Proceedings of 28th Annual Network and Distributed System Security Symposium (NDSS), 2021

    https://www.ndss-symposium.org/ndss-paper/manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/
    """

    attacker_weights = flatten_weights(weights_received)

    weights_avg = torch.mean(attacker_weights, 0)

    # Smooth benign model updates
    weights_avg = smoothing("benign", weights_avg)

    # Generate perturbation vectors (Inverse unit vector by default)
    if dev_type == "unit_vec":
        # Inverse unit vector
        perturbation_vector = weights_avg / torch.norm(weights_avg)
    elif dev_type == "sign":
        # Inverse sign
        perturbation_vector = torch.sign(weights_avg)
    elif dev_type == "std":
        # Inverse standard deviation
        perturbation_vector = torch.std(attacker_weights, 0)

    # Importance pruning
    sali_indicators_vector = compute_sali_indicator()
    perturbation_vector = perturbation_vector * sali_indicators_vector

    # Calculate the minimal sum of squared distances of benign update from the other benign updates
    min_sum_distance = torch.tensor([0])
    for attacker_weight in attacker_weights:
        distance = torch.norm((attacker_weights - attacker_weight), dim=1) ** 2
        min_sum_distance = torch.min(min_sum_distance, torch.sum(distance))

    # Search for lambda
    lambda_value = torch.Tensor([50.0]).float()
    threshold = 1e-5
    lambda_step = lambda_value
    lambda_succ = 0

    while torch.abs(lambda_succ - lambda_value) > threshold:
        poison_value = weights_avg - lambda_value * perturbation_vector
        distance = torch.norm((attacker_weights - poison_value), dim=1) ** 2
        score = torch.sum(distance)

        if score <= min_sum_distance:
            lambda_succ = lambda_value
            lambda_value = lambda_value + lambda_step / 2
        else:
            lambda_value = lambda_value - lambda_step / 2

        lambda_step = lambda_step / 2

    poison_value = weights_avg - lambda_succ * perturbation_vector

    # Smooth poison value
    poison_value = smoothing("poisoned", poison_value)

    # perform model poisoning
    weights_poisoned = perform_model_poisoning(weights_received, poison_value)
    logging.info(f"Finished Min-Sum model poisoning attack (with Oblivion).")
    return weights_poisoned


def compute_lambda(attacker_weights, global_model_last_round, num_attackers):
    """Compute the lambda value for fang's attack."""
    distances = []
    num_benign_clients, d = (
        attacker_weights.shape
    )  # impractical, not sure how many benign clients are included.

    for weight in attacker_weights:
        distance = torch.norm((attacker_weights - weight), dim=1)
        distances = (
            distance[None, :]
            if not len(distances)
            else torch.cat((distances, distance[None, :]), 0)
        )

    distances[distances == 0] = 10000
    distances = torch.sort(distances, dim=1)[0]
    scores = torch.sum(distances[:, : num_benign_clients - 2 - num_attackers], dim=1)
    score_min = torch.min(scores)

    # Calculate lambda
    term_1 = score_min / (
        (num_benign_clients - num_attackers - 1) * torch.sqrt(torch.Tensor([d]))[0]
    )
    max_wre_dist = (
        torch.max(torch.norm((attacker_weights - global_model_last_round), dim=1))
        / (torch.sqrt(torch.Tensor([d]))[0])
    )
    lambda_value = term_1 + max_wre_dist

    return lambda_value


def multi_krum(attacker_weights, num_attackers, multi_k=False):
    """multi krum defence method in secure server aggregation"""
    candidates = []
    candidate_indices = []
    remaining_updates = attacker_weights
    all_indices = np.arange(len(attacker_weights))

    while len(remaining_updates) > 2 * num_attackers + 2:
        distances = []
        for update in remaining_updates:
            distance = torch.norm((remaining_updates - update), dim=1) ** 2
            distances = (
                distance[None, :]
                if not len(distances)
                else torch.cat((distances, distance[None, :]), 0)
            )

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(
            distances[:, : len(remaining_updates) - 2 - num_attackers], dim=1
        )
        indices = torch.argsort(scores)[: len(remaining_updates) - 2 - num_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = (
            remaining_updates[indices[0]][None, :]
            if not len(candidates)
            else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        )
        remaining_updates = torch.cat(
            (remaining_updates[: indices[0]], remaining_updates[indices[0] + 1 :]), 0
        )
        if not multi_k:
            break
    weights_avg = torch.mean(candidates, dim=0)
    return weights_avg, np.array(candidate_indices)


def fang_attack(weights_received):
    """
    Attack: Fang's attack

    Reference:

    Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning," in Proceedings of USENIX Security Symposium, 2020.

    https://arxiv.org/pdf/1911.11815.pdf
    """
    num_attackers = len(Config().clients.attacker_ids)

    attacker_weights = flatten_weights(weights_received)

    weights_avg = torch.mean(attacker_weights, 0)
    global_model_last_round = weights_avg  # ?
    lambda_value = compute_lambda(
        attacker_weights, global_model_last_round, num_attackers
    )

    # Search for lambda and calculate the poison value
    threshold = 1e-5
    lambda_decay = 0.5
    deviation = torch.sign(weights_avg)
    poison_value = []

    while lambda_value > threshold:
        poison_value = -lambda_value * deviation
        poison_values = torch.stack([poison_value] * num_attackers)
        poison_values = torch.cat((poison_values, attacker_weights), 0)

        weights_avg, krum_candidate = multi_krum(
            poison_values, num_attackers, multi_k=False
        )
        if krum_candidate < num_attackers:
            # Perform model poisoning
            weights_poisoned = perform_model_poisoning(weights_received, poison_value)
            return weights_poisoned
        else:
            poison_value = []

        lambda_value *= lambda_decay

    if not len(poison_value):
        poison_value = global_model_last_round - lambda_value * deviation

    # perform model poisoning
    weights_poisoned = perform_model_poisoning(weights_received, poison_value)
    logging.info(f"Finished Fang model poisoning attack.")
    return weights_poisoned


registered_attacks = {
    "LIE": lie_attack,
    "Oblivion-lie": oblivion_lie_attack,
    "Min-Max": min_max_attack,
    "Oblivion-minmax": oblivion_min_max_attack,
    "Min-Sum": min_sum_attack,
    "Oblivion-minsum": oblivion_min_sum_attack,
    "Fang": fang_attack,
    "Lambda_attack": lambda_attack,
    "Gassian_attack": gassian_attack,
}
