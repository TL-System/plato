import copy
from collections import OrderedDict

from plato.algorithms import fedavg


class Algorithm(fedavg.Algorithm):
    """The federated learning algorithm for clustered unlearning, used by the server."""

    def __init__(self, trainer=None):
        super().__init__(trainer)

        # A dictionary that maps cluster IDs to their respective models
        self.models = {}

        # A dictionary that maps client IDs to the cluster IDs
        self.clusters = None

    def init_clusters(self, clusters):
        """Initialize the dictionary that maps cluster IDs to client IDs."""
        self.clusters = clusters

    def extract_weights(self, model=None, client_id=None):
        """Extract weights from the model."""
        if client_id is None:
            return super().extract_weights(model=model)
        else:
            cluster_id = self.clusters[client_id]

            if model is None:
                if cluster_id in self.models:
                    return self.models[cluster_id].cpu().state_dict()
                else:
                    return self.model.cpu().state_dict()
            else:
                return model.cpu().state_dict()

    def load_weights(self, weights, cluster_id=None):
        """Load the model weights passed in as a parameter."""
        if cluster_id is None:
            super().load_weights(weights)
        else:
            # Load into a particular cluster on the server
            if cluster_id not in self.models:
                self.models[cluster_id] = copy.deepcopy(self.trainer.model)

            self.models[cluster_id].load_state_dict(weights, strict=True)

    def update_weights(self, deltas, cluster_id=None):
        """Update the existing model weights."""
        if cluster_id is None:
            return super().update_weights(deltas)
        else:
            # Update the weights for a particular cluster
            baseline_weights = self.extract_weights(
                client_id=self.get_client_id(cluster_id)
            )

            updated_weights = OrderedDict()
            for name, weight in baseline_weights.items():
                updated_weights[name] = weight + deltas[name]

            return updated_weights

    def get_client_id(self, cluster_id):
        """Retrieving the corresponding client ID for a particular cluster ID."""
        for client_id, cluster in self.clusters.items():
            if cluster == cluster_id:
                return client_id

    def compute_weight_deltas(
        self, baseline_weights, weights_received, cluster_id=None
    ):
        """Extract the weights received from a client and compute the deltas."""
        # Extract baseline model weights
        if cluster_id is None:
            return super().compute_weight_deltas(baseline_weights, weights_received)
        else:
            baseline_weights = self.extract_weights(
                client_id=self.get_client_id(cluster_id)
            )

            # Calculate updates from the received weights
            deltas = []
            for weight in weights_received:
                delta = OrderedDict()
                for name, current_weight in weight.items():
                    baseline = baseline_weights[name]

                    # Calculate update
                    _delta = current_weight - baseline
                    delta[name] = _delta
                deltas.append(delta)

            return deltas
