"""
The registry that contains all available model poisoning attacks in federated learning.

Having a registry of all available classes is convenient for retrieving an instance based
on a configuration at run-time.
"""

import torch
import logging
from plato.config import Config
from scipy.stats import norm
from collections import OrderedDict
import numpy as np


def get():
    """Get an attack for malicious clients based on the configuration file."""
    attack_type = (
        Config().clients.attack_type
        if hasattr(Config().clients, "attack_type")
        else None
    )

    if attack_type is None:
        logging.info(f"No attack is applied.")
        return lambda x: x

    if attack_type in registered_attacks:
        registered_attack = registered_attacks[attack_type]
        logging.info(f"Clients perform {attack_type} attack.")
        return registered_attack

    raise ValueError(f"No such attack: {attack_type}")


def perform_model_poisoning(weights_received, poison_value):
    # Poison the reveiced weights based on calculated poison value.
    weights_poisoned = []
    for weight_received in weights_received:
        start_index = 0
        weight_poisoned = OrderedDict()

        for name, weight in weight_received.items():
            weight_poisoned[name] = poison_value[
                start_index : start_index + len(weight.view(-1))
            ].reshape(weight.shape)
            start_index += len(weight.view(-1))

        weights_poisoned.append(weight_poisoned)
    return weights_poisoned


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


def lie_attack(weights_received):
    """
    Attack name: Little is enough

    Reference:

    Baruch et al., "A little is enough: Circumventing defenses for distributed learning," in Proceedings of Advances in Neural Information Processing Systems (NeurIPS) 2019.

    https://proceedings.neurips.cc/paper_files/paper/2019/file/ec1c59141046cd1866bbbcdfb6ae31d4-Paper.pdf
    """

    total_clients = Config().clients.total_clients
    num_attackers = len(Config().clients.attacker_ids)

    attacker_weights = flatten_weights(weights_received)

    # Calculate perturbation range
    s_value = total_clients / 2 + 1 - num_attackers
    possibility = (total_clients - s_value) / total_clients
    z_value = norm.cdf(possibility)

    weights_avg = torch.mean(attacker_weights, dim=0)
    weights_std = torch.std(attacker_weights, dim=0)

    # Calculate poisoning model
    poison_value = weights_avg + z_value * weights_std
    # Perform model poisoning
    weights_poisoned = perform_model_poisoning(weights_received, poison_value)
    logging.info(f"Finished LIE model poisoning attack.")
    return weights_poisoned


def min_max_attack(weights_received, dev_type="unit_vec"):
    """
    Attack: Min-Max

    Reference:

    Shejwalkar et al., “Manipulating the Byzantine: Opti- mizing model poisoning attacks and defenses for federated learning,” in Proceedings of 28th Annual Network and Distributed System Security Symposium (NDSS), 2021

    https://www.ndss-symposium.org/ndss-paper/manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/
    """
    attacker_weights = flatten_weights(weights_received)

    weights_avg = torch.mean(attacker_weights, 0)
    global_model_last_round = weights_avg

    if dev_type == "unit_vec":
        deviation = global_model_last_round / torch.norm(
            global_model_last_round
        )  # unit vector, dir opp to good dir
    elif dev_type == "sign":
        deviation = torch.sign(global_model_last_round)
    elif dev_type == "std":
        deviation = torch.std(attacker_weights, 0)

    lambda_value = torch.Tensor([50.0]).float()
    threshold = 1e-5
    lambda_fail = lambda_value
    lambda_succ = 0

    # Search for maximal distance
    distances = []
    for attacker_weight in attacker_weights:
        distance = torch.norm((attacker_weights - attacker_weight), dim=1) ** 2
        distances = (
            distance[None, :]
            if not len(distances)
            else torch.cat((distances, distance[None, :]), 0)
        )

    max_distance = torch.max(distances)
    del distances

    # Search for lambda
    while torch.abs(lambda_succ - lambda_value) > threshold:
        poison_value = global_model_last_round - lambda_value * deviation
        distance = torch.norm((attacker_weights - poison_value), dim=1) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            lambda_succ = lambda_value
            lambda_value = lambda_value + lambda_fail / 2
        else:
            lambda_value = lambda_value - lambda_fail / 2

        lambda_fail = lambda_fail / 2

    poison_value = global_model_last_round - lambda_succ * deviation

    # perform model poisoning
    weights_poisoned = perform_model_poisoning(weights_received, poison_value)
    logging.info(f"Finished Min-Max model poisoning attack.")
    return weights_poisoned


def min_sum_attack(weights_received, dev_type="unit_vec"):
    """
    Attack: Min-Max

    Reference:

    Shejwalkar et al., “Manipulating the Byzantine: Opti- mizing model poisoning attacks and defenses for federated learning,” in Proceedings of 28th Annual Network and Distributed System Security Symposium (NDSS), 2021

    https://www.ndss-symposium.org/ndss-paper/manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/
    """

    attacker_weights = flatten_weights(weights_received)

    weights_avg = torch.mean(attacker_weights, 0)

    global_model_last_round = weights_avg

    if dev_type == "unit_vec":
        deviation = global_model_last_round / torch.norm(
            global_model_last_round
        )  # unit vector, dir opp to good dir
    elif dev_type == "sign":
        deviation = torch.sign(global_model_last_round)
    elif dev_type == "std":
        deviation = torch.std(attacker_weights, 0)

    # Calculate minimal score
    distances = []
    for attacker_weight in attacker_weights:
        distance = torch.norm((attacker_weights - attacker_weight), dim=1) ** 2
        distances = (
            distance[None, :]
            if not len(distances)
            else torch.cat((distances, distance[None, :]), 0)
        )

    scores = torch.sum(distances, dim=1)
    score_min = torch.min(scores)
    del distances

    # Search for lambda
    lambda_value = torch.Tensor([50.0]).float()
    threshold = 1e-5
    lambda_fail = lambda_value
    lambda_succ = 0

    while torch.abs(lambda_succ - lambda_value) > threshold:
        poison_value = global_model_last_round - lambda_value * deviation
        distance = torch.norm((attacker_weights - poison_value), dim=1) ** 2
        score = torch.sum(distance)

        if score <= score_min:
            lambda_succ = lambda_value
            lambda_value = lambda_value + lambda_fail / 2
        else:
            lambda_value = lambda_value - lambda_fail / 2

        lambda_fail = lambda_fail / 2

    poison_value = global_model_last_round - lambda_succ * deviation

    # perform model poisoning
    weights_poisoned = perform_model_poisoning(weights_received, poison_value)
    logging.info(f"Finished Min-Sum model poisoning attack.")
    return weights_poisoned


def compute_lambda(attacker_weights, global_model_last_round, num_attackers):
    """Compute the lambda value for fang's attack."""
    distances = []
    num_benign_clients, d = attacker_weights.shape  # ? total - num_attacker?

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
    max_wre_dist = torch.max(
        torch.norm((attacker_weights - global_model_last_round), dim=1)
    ) / (torch.sqrt(torch.Tensor([d]))[0])
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
    global_model_last_round = weights_avg
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
        logging.info(f"poison_value: %s", poison_value)
        logging.info(f"poison_values: %s", poison_values)

        weights_avg, krum_candidate = multi_krum(
            poison_values, num_attackers, multi_k=False
        )
        logging.info(f"krum_candidate: ", krum_candidate)
        if krum_candidate < num_attackers:
            # perform model poisoning
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
    "Min-Max": min_max_attack,
    "Min-Sum": min_sum_attack,
    "Fang": fang_attack,
}
