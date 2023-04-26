import torch
import logging
from plato.config import Config
from scipy.stats import norm


async def lie_attack(self, updates):
    """Little is enough"""
    """     LIE attack   """
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
