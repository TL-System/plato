"""
A federated semi-supervised learning server using FedMatch.

Reference:

Jeong et al., "Federated Semi-supervised learning with inter-client consistency and
disjoint learning", in the Proceedings of ICLR 2021.

https://arxiv.org/pdf/2006.12097.pdf
"""
import numpy as np
from scipy.stats.stats import mode
from plato.servers import fedavg
from scipy import spatial
from scipy.stats import truncnorm
from plato.models import de_lenet5_decomposed
import torch


class Server(fedavg.Server):
    """A federated learning server using the FedMatch algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        self.num_helpers = 4
        self.helper = [False] * (self.num_helpers + 1)
        mu, std, lower, upper = 125, 125, 0, 255
        self.gauss_samples = (truncnorm(
            (lower - mu) / std, (upper - mu) / std, loc=mu, scale=std).rvs(
                (1, 1, 28, 28)))  #32, 32, 3)))  #/ 255

    def extract_client_updates(self, updates):
        """ Extract the model weights and control variates from clients updates. """

        weights_received = [payload for (__, payload) in updates]
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
        if self.helper[selected_client_id - 1] is True:

            helpers = self.helpers[selected_client_id]
            print("Select helpers for client #", selected_client_id)

            return [payload, helpers]
        self.helper[selected_client_id - 1] = True

        return payload

    def compute_similarity(self, models_received, client_ids):
        "compute similarity among clients"

        # initialize vector as an empty dictionary with length of client_ids
        self.models_dict = {}
        out_list = []

        for cid, model_dict in zip(client_ids, models_received):
            model = de_lenet5_decomposed.Model()
            model.load_state_dict(model_dict)
            out = model(torch.from_numpy(self.gauss_samples).float())
            self.models_dict[cid] = np.squeeze(out)
            out_list.append(np.squeeze(out.detach().numpy()))

        self.tree = spatial.KDTree(out_list)

    def find_helpers(self, client_id, models_received):
        print("The coming clients are: ", client_id)

        helper_dict = {}
        for id in client_id:
            distances, similiar_model_ids = self.tree.query(
                self.models_dict[id].detach().numpy(), self.num_helpers + 1)
            similiar_model_ids = similiar_model_ids + 1
            #print("Current id is: ", id)

            sim_ids = similiar_model_ids.tolist()
            sim_ids.remove(id)
            #print("Sim_ids is : ", sim_ids)
            weights = []
            for sim_id in sim_ids:
                weights.append(models_received[client_id.index(sim_id)])
            helper_dict[id] = weights
        return helper_dict
