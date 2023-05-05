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


def lie_attack(weights_received):
    """
    Attack name: Little is enough

    Reference:

    Baruch et al., "A little is enough: Circumventing defenses for distributed learning," in Proceedings of Advances in Neural Information Processing Systems (NeurIPS) 2019.

    https://proceedings.neurips.cc/paper_files/paper/2019/file/ec1c59141046cd1866bbbcdfb6ae31d4-Paper.pdf
    """

    total_clients = Config().clients.total_clients
    num_attackers = len(Config().clients.attacker_ids)

    attacker_grads = []

    for weight_received in weights_received:
        delta_vector = []
        for name in weight_received.keys():
            delta_vector = (
                weight_received[name].view(-1)
                if not len(delta_vector)
                else torch.cat((delta_vector, weight_received[name].view(-1)))
            )

        attacker_grads = (
            delta_vector[None, :]
            if not len(attacker_grads)
            else torch.cat((attacker_grads, delta_vector[None, :]), 0)
        )

    s_value = total_clients / 2 + 1 - num_attackers
    possibility = (total_clients - s_value) / total_clients
    z_value = norm.cdf(possibility)

    # calculate poisoning model
    update_avg = torch.mean(attacker_grads, dim=0)
    update_std = torch.std(attacker_grads, dim=0)

    mal_update = update_avg + z_value * update_std

    # perform model poisoning
    weights_poisoned = []
    for weight_received in weights_received:
        start_index = 0
        weight_poisoned = OrderedDict()

        for name, weight in weight_received.items():

            weight_poisoned[name] = mal_update[
                start_index : start_index + len(weight.view(-1))
            ].reshape(weight.shape)
            start_index += len(weight.view(-1))

        weights_poisoned.append(weight_poisoned)

    return weights_poisoned


def min_max(weights_received, dev_type="unit_vec"):
    """
    Attack: Min-Max

    Reference:

    Shejwalkar et al., “Manipulating the Byzantine: Opti- mizing model poisoning attacks and defenses for federated learning,” in Proceedings of 28th Annual Network and Distributed System Security Symposium (NDSS), 2021

    https://www.ndss-symposium.org/ndss-paper/manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/
    """

    attacker_grads = []
    for weight_received in weights_received:
        delta_vector = []
        for name in weight_received.keys():
            delta_vector = (
                weight_received[name].view(-1)
                if not len(delta_vector)
                else torch.cat((delta_vector, weight_received[name].view(-1)))
            )

        attacker_grads = (
            delta_vector[None, :]
            if not len(attacker_grads)
            else torch.cat((attacker_grads, delta_vector[None, :]), 0)
        )

    agg_grads = torch.mean(attacker_grads, 0)

    model_re = agg_grads

    if dev_type == "unit_vec":
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == "sign":
        deviation = torch.sign(model_re)
    elif dev_type == "std":
        deviation = torch.std(attacker_grads, 0)

    lamda = torch.Tensor([50.0]).float()
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    distances = []

    for update in attacker_grads:
        distance = torch.norm((attacker_grads - update), dim=1) ** 2
        distances = (
            distance[None, :]
            if not len(distances)
            else torch.cat((distances, distance[None, :]), 0)
        )

    max_distance = torch.max(distances)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = model_re - lamda * deviation
        distance = torch.norm((attacker_grads - mal_update), dim=1) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = model_re - lamda_succ * deviation

    # perform model poisoning
    weights_poisoned = []
    for weight_received in weights_received:
        start_index = 0
        weight_poisoned = OrderedDict()

        for name, weight in weight_received.items():
            weight_poisoned[name] = mal_update[
                start_index : start_index + len(weight.view(-1))
            ].reshape(weight.shape)
            start_index += len(weight.view(-1))

        weights_poisoned.append(weight_poisoned)

    return weights_poisoned


def min_sum(weights_received, dev_type="unit_vec"):
    """
    Attack: Min-Max

    Reference:

    Shejwalkar et al., “Manipulating the Byzantine: Opti- mizing model poisoning attacks and defenses for federated learning,” in Proceedings of 28th Annual Network and Distributed System Security Symposium (NDSS), 2021

    https://www.ndss-symposium.org/ndss-paper/manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/
    """

    attacker_grads = []
    for weight_received in weights_received:
        delta_vector = []
        for name in weight_received.keys():
            delta_vector = (
                weight_received[name].view(-1)
                if not len(delta_vector)
                else torch.cat((delta_vector, weight_received[name].view(-1)))
            )
        attacker_grads = (
            delta_vector[None, :]
            if not len(attacker_grads)
            else torch.cat((attacker_grads, delta_vector[None, :]), 0)
        )

    agg_grads = torch.mean(attacker_grads, 0)

    model_re = agg_grads

    if dev_type == "unit_vec":
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == "sign":
        deviation = torch.sign(model_re)
    elif dev_type == "std":
        deviation = torch.std(attacker_grads, 0)

    lamda = torch.Tensor([50.0]).float()

    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    distances = []

    for update in attacker_grads:
        distance = torch.norm((attacker_grads - update), dim=1) ** 2
        distances = (
            distance[None, :]
            if not len(distances)
            else torch.cat((distances, distance[None, :]), 0)
        )

    scores = torch.sum(distances, dim=1)
    min_score = torch.min(scores)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = model_re - lamda * deviation
        distance = torch.norm((attacker_grads - mal_update), dim=1) ** 2
        score = torch.sum(distance)

        if score <= min_score:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = model_re - lamda_succ * deviation

    # perform model poisoning
    weights_poisoned = []
    for weight_received in weights_received:
        start_index = 0
        weight_poisoned = OrderedDict()

        for name, weight in weight_received.items():
            weight_poisoned[name] = mal_update[
                start_index : start_index + len(weight.view(-1))
            ].reshape(weight.shape)
            start_index += len(weight.view(-1))

        weights_poisoned.append(weight_poisoned)
    return weights_poisoned


def compute_lambda(all_updates, model_re, n_attackers):
    """Compute lambda value for fang's attack"""

    distances = []
    n_benign, d = all_updates.shape
    for update in all_updates:
        distance = torch.norm((all_updates - update), dim=1)
        distances = (
            distance[None, :]
            if not len(distances)
            else torch.cat((distances, distance[None, :]), 0)
        )

    distances[distances == 0] = 10000
    distances = torch.sort(distances, dim=1)[0]
    scores = torch.sum(distances[:, : n_benign - 2 - n_attackers], dim=1)
    min_score = torch.min(scores)
    term_1 = min_score / (
        (n_benign - n_attackers - 1) * torch.sqrt(torch.Tensor([d]))[0]
    )
    max_wre_dist = torch.max(torch.norm((all_updates - model_re), dim=1)) / (
        torch.sqrt(torch.Tensor([d]))[0]
    )

    return term_1 + max_wre_dist


def multi_krum(all_updates, n_attackers, multi_k=False):
    """multi krum defence method in secure server aggregation"""
    candidates = []
    candidate_indices = []
    remaining_updates = all_updates
    all_indices = np.arange(len(all_updates))

    while len(remaining_updates) > 2 * n_attackers + 2:
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
            distances[:, : len(remaining_updates) - 2 - n_attackers], dim=1
        )
        indices = torch.argsort(scores)[: len(remaining_updates) - 2 - n_attackers]

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
    aggregate = torch.mean(candidates, dim=0)
    return aggregate, np.array(candidate_indices)


def fang_attack(weights_received):
    """
    Attack: Fang's attack

    Reference:

    Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning," in Proceedings of USENIX Security Symposium, 2020.

    https://arxiv.org/pdf/1911.11815.pdf
    """
    num_attackers = len(Config().clients.attacker_ids)
    attacker_grads = []

    for weight_received in weights_received:
        delta_vector = []
        for name in weight_received.keys():
            delta_vector = (
                weight_received[name].view(-1)
                if not len(delta_vector)
                else torch.cat((delta_vector, weight_received[name].view(-1)))
            )
        attacker_grads = (
            delta_vector[None, :]
            if not len(attacker_grads)
            else torch.cat((attacker_grads, delta_vector[None, :]), 0)
        )

    agg_grads = torch.mean(attacker_grads, 0)
    model_re = agg_grads
    deviation = torch.sign(agg_grads)
    lamda = compute_lambda(attacker_grads, model_re, num_attackers)

    threshold = 1e-5
    mal_update = []

    while lamda > threshold:
        mal_update = -lamda * deviation
        mal_updates = torch.stack([mal_update] * num_attackers)
        mal_updates = torch.cat((mal_updates, attacker_grads), 0)

        agg_grads, krum_candidate = multi_krum(
            mal_updates, num_attackers, multi_k=False
        )
        if krum_candidate < num_attackers:
            # perform model poisoning
            weights_poisoned = []
            for weight_received in weights_received:
                start_index = 0
                weight_poisoned = OrderedDict()

                for name, weight in weight_received.items():
                    weight_poisoned[name] = mal_update[
                        start_index : start_index + len(weight.view(-1))
                    ].reshape(weight.shape)
                    start_index += len(weight.view(-1))

                weights_poisoned.append(weight_poisoned)
            return weights_poisoned
        else:
            mal_update = []

        lamda *= 0.5

    if not len(mal_update):
        mal_update = model_re - lamda * deviation

    # perform model poisoning
    weights_poisoned = []
    for weight_received in weights_received:
        start_index = 0
        weight_poisoned = OrderedDict()

        for name, weight in weight_received.items():
            weight_poisoned[name] = mal_update[
                start_index : start_index + len(weight.view(-1))
            ].reshape(weight.shape)
            start_index += len(weight.view(-1))

        weights_poisoned.append(weight_poisoned)
    return weights_poisoned


registered_attacks = {
    "LIE": lie_attack,
    "Min-Max": min_max,
    "Min-Sum": min_sum,
    "Fang": fang_attack,
}
