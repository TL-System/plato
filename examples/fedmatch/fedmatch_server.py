"""
A federated semi-supervised learning server using FedMatch.

Reference:

Jeong et al., "Federated Semi-supervised learning with inter-client consistency and
disjoint learning", in the Proceedings of ICLR 2021.

https://arxiv.org/pdf/2006.12097.pdf
"""
import numpy as np
from plato.servers import fedavg
from scipy import spatial
from scipy.stats import truncnorm


class Server(fedavg.Server):
    """A federated learning server using the FedMatch algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        self.num_helpers = 5
        self.helper_flag = 0
        mu, std, lower, upper = 125, 125, 0, 255
        self.gauss_samples = (truncnorm(
            (lower - mu) / std, (upper - mu) / std, loc=mu, scale=std).rvs(
                (1, 32, 32, 3))) / 255

    def extract_client_updates(self, updates):
        """ Extract the model weights and control variates from clients updates. """
        weights_received = [payload[0] for (__, payload) in updates]

        self.control_variates_received = [
            payload[1] for (__, payload) in updates
        ]

        return self.algorithm.compute_weight_updates(weights_received)

    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using FedMatch."""
        # sigmas and psis received from clients
        models_received = [payload for (__, payload) in updates]
        # clients' ids received
        client_ids = [report.client_id for (report, __) in updates]
        # compute similarity for clients
        self.compute_similarity(models_received, client_ids)
        # find helpers for each client
        self.helpers = self.find_helpers(
            client_ids, models_received)  #return is a dictionary

        # later do averaging
        update = await super().federated_averaging(updates)

        return update

    def customize_server_payload(self, payload, selected_client_id):
        "Add helpers models to payload for each client"
        helpers = self.helpers[selected_client_id]

        return [payload, helpers]

    def compute_similarity(self, models_received, client_ids):
        "compute similarity among clients"

        # initialize vector as an empty dictionary with length of client_ids
        self.models_dict = {}

        for cid, model in zip(client_ids, models_received):
            self.models_dict[cid] = np.squeeze(model(self.gauss_samples))
        self.tree = spatial.KDTree(list(self.models_dict.values()))
        """
        for cid, update in enumerate(updates_all):
            for model_weight in update:
                self.cid_to_vectors[cid] = np.squeeze(rmodel(self.rgauss))  #
        self.vid_to_cid = list(self.cid_to_vectors.keys())
        self.vectors = list(self.cid_to_vectors.values())
        self.tree = spatial.KDTree(self.vectors)
        """

    def find_helpers(self, client_id, models_received):
        helper_dict = {}
        for id in client_id:
            distances, similiar_model_ids = self.tree.query(
                self.models_dict[id], self.num_helpers + 1)
            similiar_model_ids = similiar_model_ids[1:]  # remove itself
            weights = []
            for sim_id in similiar_model_ids:
                weights.append(models_received[client_id.index(sim_id)])
            helper_dict[id] = weights
        return helper_dict
        """

        cout = self.cid_to_vectors[client_id]
        sims = self.tree.query(cout, self.args.num_helpers + 1)
        hids = []
        weights = []
        for vid in sims[1]:
            selected_cid = self.vid_to_cid[vid]
            if selected_cid == cid:
                continue
            w = self.cid_to_weights[selected_cid]
            if self.args.scenario == 'labels-at-client':
                half = len(w) // 2
                w = w[half:]
            weights.append(w)
            hids.append(selected_cid)
        return weights[:self.num_helpers]
        """
