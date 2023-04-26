import torch
import logging
from plato.config import Config
from scipy.stats import norm


def get():

    attack_type = (
        Config().clients.attack_type
        if hasattr(Config().clients, "attack_type")
        else None
    )

    if attack_type is None:
        logging.info("No attack is applied.")
        return lambda x: x

    if attack_type in registered_attacks:
        registered_attack = registered_attacks[attack_type]
        return registered_attack

    raise ValueError(f"No such attack: {attack_type}")


def lie_attack(self, updates):
    """Little is enough"""
    """https://proceedings.neurips.cc/paper_files/paper/2019/file/ec1c59141046cd1866bbbcdfb6ae31d4-Paper.pdf """

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

    attacker_grads = []
    attacker_ids = []
    tmp_index = 0
    for client_id in clients_id:
        if (client_id <= self.n_attackers) and (client_id != 0):
            delta_vector = []
            attacker_ids.append(client_id)
            delta_received = deltas_received[tmp_index]
            for name in name_list:
                delta_vector = (
                    delta_received[name].view(-1)
                    if not len(delta_vector)
                    else torch.cat((delta_vector, delta_received[name].view(-1)))
                )
            attacker_grads = (
                delta_vector[None, :]
                if not len(attacker_grads)
                else torch.cat((attacker_grads, delta_vector[None, :]), 0)
            )
        tmp_index = tmp_index + 1

    s_value = n_clients / 2 + 1 - n_attackers
    possibility = (n_clients - s_value) / n_clients
    z_value = norm.cdf(possibility)

    # all_updates: know bengin clients' updates
    # attacker_grads: unknown bengin clients' updates
    if Config().algorithm.partial:
        update_avg = torch.mean(attacker_grads, dim=0)
        update_std = torch.std(attacker_grads, dim=0)
    else:
        update_avg = torch.mean(all_updates, dim=0)
        update_std = torch.std(all_updates, dim=0)

    mal_update = update_avg + z_value * update_std
    self.renew_malicious_updates(deltas_received, clients_id, mal_update, attacker_ids)

    return mal_update


def min_max(self, updates, dev_type="unit_vec"):
    """
    min_max attack
    Two options:
        1. benign clients' updates are known to the attacker
        2. benign clients' updates are unknown to the attacker
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
    logging.info("[%s] n_clients: %d", self, n_clients)
    logging.info("[%s] n_attackers: %d", self, n_attackers)

    attacker_grads = []
    attacker_ids = []
    tmp_index = 0
    for client_id in clients_id:
        if (client_id <= self.n_attackers) and (client_id != 0):
            delta_vector = []
            attacker_ids.append(client_id)
            delta_received = deltas_received[tmp_index]
            for name in name_list:
                delta_vector = (
                    delta_received[name].view(-1)
                    if not len(delta_vector)
                    else torch.cat((delta_vector, delta_received[name].view(-1)))
                )
            attacker_grads = (
                delta_vector[None, :]
                if not len(attacker_grads)
                else torch.cat((attacker_grads, delta_vector[None, :]), 0)
            )
        tmp_index = tmp_index + 1

    if Config().algorithm.partial:
        agg_grads = torch.mean(attacker_grads, 0)
    else:
        agg_grads = torch.mean(all_updates, 0)

    model_re = agg_grads

    if dev_type == "unit_vec":
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == "sign":
        deviation = torch.sign(model_re)
    elif dev_type == "std":
        if Config().algorithm.partial:
            deviation = torch.std(attacker_grads, 0)
        else:
            deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([50.0]).float()
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    distances = []
    # all_updates: know bengin clients' updates
    # attacker_grads: unknown bengin clients' updates
    if Config().algorithm.partial:
        for update in attacker_grads:
            distance = torch.norm((attacker_grads - update), dim=1) ** 2
            distances = (
                distance[None, :]
                if not len(distances)
                else torch.cat((distances, distance[None, :]), 0)
            )
    else:
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = (
                distance[None, :]
                if not len(distances)
                else torch.cat((distances, distance[None, :]), 0)
            )

    max_distance = torch.max(distances)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = model_re - lamda * deviation
        distance = torch.norm((all_updates - mal_update), dim=1) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = model_re - lamda_succ * deviation
    self.renew_malicious_updates(deltas_received, clients_id, mal_update, attacker_ids)
    return mal_update


registered_attacks = {"LIE": lie_attack, "Min-Max": min_max}
