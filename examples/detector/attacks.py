import torch
import logging
from plato.config import Config
from scipy.stats import norm
from collections import OrderedDict


def get():

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
    """Little is enough"""
    """https://proceedings.neurips.cc/paper_files/paper/2019/file/ec1c59141046cd1866bbbcdfb6ae31d4-Paper.pdf """

    """
    deltas_received = self.compute_weight_deltas(updates)
    reports = [report for (__, report, __, __) in updates]
    clients_id = [client for (client, __, __, __) in updates]

    name_list = []
    for name, delta in deltas_received[0].items():
        name_list.append(name)

    all_updates = []
    for i, delta_received in enumerate(deltas_received):
        delta_vector = []
        for name in name_list:
            delta_vector = (
                delta_received[name].view(-1)
                if not len(delta_vector)
                else torch.cat((delta_vector, delta_received[name].view(-1)))
            )
        all_updates = (
            delta_vector[None, :]
            if not len(all_updates)
            else torch.cat((all_updates, delta_vector[None, :]), 0)
        )

    n_clients = all_updates.shape[0]
    n_attackers = self.attacker_number
    logging.info("[%s] n_clients: %d", self, n_clients)
    logging.info("[%s] n_attackers: %d", self, n_attackers)
    """

    total_clients = Config().clients.total_clients
    num_attackers = len(Config().clients.attacker_ids)

    attacker_grads = []
    # attacker_ids = []

    # tmp_index = 0
    for weight_received in weights_received:
        # if (client_id <= self.n_attackers) and (client_id != 0):
        delta_vector = []
        # attacker_ids.append(client_id)
        # delta_received = deltas_received[tmp_index]
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
    min_max attack
    Two options:
        1. benign clients' updates are known to the attacker
        2. benign clients' updates are unknown to the attacker
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
    min_sum attack
    Two options:
        1. benign clients' updates are known to the attacker
        2. benign clients' updates are unknown to the attacker
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


registered_attacks = {"LIE": lie_attack, "Min-Max": min_max, "Min-Sum": min_sum}
