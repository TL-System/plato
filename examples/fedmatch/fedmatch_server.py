"""
A federated semi-supervised learning server using FedMatch.
Reference:
Jeong et al., "Federated Semi-supervised learning with inter-client consistency & disjoint learning", in the Proceedings of ICLR 2021.
https://arxiv.org/pdf/2006.12097.pdf 
"""
from scipy import spatial
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the FedMatch algorithm."""
    def __init__(self):
        super().__init__()
        self.num_helpers = 5
        self.helper_flag = 0

    def extract_client_updates(self, updates):
        """ Extract the model weights and control variates from clients updates. """
        weights_received = [payload[0] for (__, payload) in updates]

        self.control_variates_received = [
            payload[1] for (__, payload) in updates
        ]

        return self.algorithm.compute_weight_updates(weights_received)

    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using FedMatch."""

        # compute similarity for clients
        self.compute_similarity(updates)

        # find helpers for each client # only deltas received as updates, so server have to compute for the models
        helpers = find_helpers(updates)

        # later do averaging
        update = await super().federated_averaging(updates)

        return update

    def customize_server_payload(self, payload, selected_client_id):
        "Add server control variates into the server payload."
        if self.helper_flag == 0:
            return payload

        else:
            helpers = self.find_helpers(selected_client_id)
            return helpers.insert(0, payload)

    def compute_similarity(self, updates):
        "compute similarity among clients"
        updates_all = updates[0] + updates[1]

        for cid, update in enumerate(updates_all):
            for model_weight in update:
                self.cid_to_vectors[cid] = np.squeeze(rmodel(self.rgauss))  #
        self.vid_to_cid = list(self.cid_to_vectors.keys())
        self.vectors = list(self.cid_to_vectors.values())
        self.tree = spatial.KDTree(self.vectors)

    def compute_similarity(self, updates):
        "compute similarity among clients and build tree for finding helpers"
        # sum up the psi and sigma from local update and get model weights of all clients
        embeded_model = []
        for cid, psi, sigma in enumerate(updates):
            weight = psi + sigma
            prediction = np.squeeze(weight(self.gauss))

        #

    def find_helpers(self, client_id):
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
