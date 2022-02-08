"""
A federated learning server with RL Agent FEI
"""
import math

import numpy as np
from plato.config import Config
from plato.utils.reinforcement_learning import simple_rl_server


class RLServer(simple_rl_server.RLServer):
    """ A federated learning server with RL Agent. """
    def __init__(self, agent, model=None, algorithm=None, trainer=None):
        super().__init__(agent, model, algorithm, trainer)
        self.local_correlations = {}
        self.last_global_grads = None
        self.corr = []
        self.smart_weighting = []

    # Overwrite RL-related methods of simple RL server
    def prep_state(self):
        """ Wrap up the state update to RL Agent. """
        state = [0] * 4
        if len(self.updates) > 0 and len(self.updates) >= len(
                self.selected_clients):
            state[0] = self.normalize_state(
                [report.num_samples for (report, __, __) in self.updates])
            state[1] = self.normalize_state(
                [report.training_time for (report, __, __) in self.updates])
            state[2] = self.normalize_state(
                [report.valuation for (report, __, __) in self.updates])
            state[3] = self.normalize_state(self.corr)
        state = np.transpose(np.round(np.array(state), 4))

        self.agent.test_accuracy = self.accuracy

        return state

    def apply_action(self):
        """ Apply action update from RL Agent to FL Env. """
        self.smart_weighting = np.array(self.agent.action)

    def extract_client_updates(self, updates):
        """ Extract the model weights and update directions from clients updates. """
        weights_received = [payload for (__, payload, __) in updates]

        # Get adaptive weighting based on both node contribution and date size
        self.corr = self.calc_corr(weights_received)

        return self.algorithm.compute_weight_updates(weights_received)

    def calc_corr(self, updates):
        """ Calculate the node contribution based on the angle
            between local gradient and global gradient.
        """
        correlations, contribs = [None] * len(updates), [None] * len(updates)

        # Update the baseline model weights
        curr_global_grads = self.process_grad(self.algorithm.extract_weights())
        if self.last_global_grads is None:
            self.last_global_grads = np.zeros(len(curr_global_grads))
        global_grads = np.subtract(curr_global_grads, self.last_global_grads)
        self.last_global_grads = curr_global_grads

        # Compute angles in radian between local and global gradients
        for i, update in enumerate(updates):
            local_grads = self.process_grad(update)
            inner = np.inner(global_grads, local_grads)
            norms = np.linalg.norm(global_grads) * np.linalg.norm(local_grads)
            correlations[i] = np.arccos(np.clip(inner / norms, -1.0, 1.0))

        for i, correlation in enumerate(correlations):
            client_id = self.selected_clients[i]
            # Update the smoothed angle for all clients
            if client_id not in self.local_correlations.keys():
                self.local_correlations[client_id] = correlation
            self.local_correlations[client_id] = (
                (self.current_round - 1) /
                self.current_round) * self.local_correlations[client_id] + (
                    1 / self.current_round) * correlation
            # Non-linear mapping to node contribution

            contribs[i] = Config().algorithm.alpha * (
                1 -
                math.exp(-math.exp(-Config().algorithm.alpha *
                                   (self.local_correlations[client_id] - 1))))

        return contribs

    @staticmethod
    def process_grad(grads):
        """ Convert gradients to a flattened 1-D array. """
        grads = list(
            dict(sorted(grads.items(), key=lambda x: x[0].lower())).values())

        flattened = grads[0]
        for i in range(1, len(grads)):
            flattened = np.append(flattened, grads[i])

        return flattened

    @staticmethod
    def normalize_state(feature):
        """Normalize/Scaling state features."""
        norm = np.linalg.norm(feature)
        return [x / norm for x in feature]
