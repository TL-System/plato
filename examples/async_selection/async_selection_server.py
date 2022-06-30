"""
A customized server with asynchronous client selection
"""
import logging

from dataclasses import replace
from plato.config import Config
from plato.servers import fedavg
from cvxopt import matrix, log, solvers, sparse
from plato.config import Config
import mosek
import numpy as np


class Server(fedavg.Server):
    """A federated learning server."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)

        self.number_of_client = Config().clients.total_clients

        # create a record on server
        self.local_gradient_bounds = 0.01 * np.ones(
            self.number_of_client
        )  # 0.01 is a hyperparameter that's used as a starting point.
        self.local_stalenesses = np.zeros(self.number_of_client)
        self.aggregate_weights = np.ones(self.number_of_client)
        self.squared_deltas_current_round = np.zeros(self.number_of_client)

    def choose_clients(self, clients_pool, clients_count):
        """ Choose a subset of the clients to participate in each round. """
        assert clients_count <= len(clients_pool)

        # Select clients based on calculated probability
        p = self.calculate_selection_probability(
            self.aggre_weights, self.local_gradient_bounds,
            self.local_stalenesses)  # arguments are np.array s

        selected_clients = np.random.choice(clients_pool,
                                            clients_count,
                                            replace=False,
                                            p=p)

        logging.info("[%s] Selected clients: %s", self, selected_clients)
        return selected_clients

    def compute_weight_deltas(self, updates):
        """Extract the model weight updates from client updates and compute local update bounds with respect to each clients id."""
        weights_received = [payload for (__, __, payload, __) in updates]
        id_received = [client_id for (client_id, __, __, __) in updates]

        # Extract baseline model weights
        baseline_weights = self.extract_weights()

        # Calculate updates from the received weights
        deltas = []
        self.squared_deltas_current_round = np.zeros(self.number_of_client)
        for client_id, weight in zip(id_received, weights_received):

            delta = OrderedDict()
            for name, current_weight in weight.items():
                baseline = baseline_weights[name]

                # Calculate update
                _delta = current_weight - baseline
                delta[name] = _delta
            deltas.append(delta)

            self.squared_deltas[client_id] = np.square(
                delta)  # may need to sum up
        return deltas  #, squared_deltas

    async def federated_averaging(self, updates):
        update = await super().federated_averaging(updates)
        # update records of local gradient norm and local staleness for corresponding client. update based on client_id
        # update here cause we have updates here!

        # Extract the deltas.
        #deltas_received, squared_deltas = self.compute_weight_deltas(updates)
        #delta_ids = [id for (id, __, __, __) in updates]
        # update record of local gradient bound.
        self.local_gradient_bounds += self.squared_deltas_current_round

        # Extract the local staleness and update the record of client staleness.
        for (id, __, __, client_staleness) in updates:
            self.local_stalenesses[
                id] = client_staleness  # replace old with new ones; can be improved using exponential moving averaging later.

        # Extract the aggre_weight
        

    def calculate_selection_probability(self, aggre_weight,
                                        local_gradient_bound, local_staleness):
        """Calculte selection probability based on the formulated geometric optimization problem
            Minimize \alpha \sum_{i=1}^N \frac{p_i^2 * G_i^2}{q_i} + A \sum_{i=1}^N q_i * \tau_i * G_i
            Subject to \sum_{i=1}{N} q_i = 1
                    q_i > 0 
            Probability Variables are q_i

        """
        # read aggre_weight from somewhere
        aggre_weight_square = np.square(self.aggre_weight)  # p_i^2
        local_gradient_bound_square = np.square(
            self.local_gradient_bound)  # G_i^2

        f1_params = alpha * np.multiply(
            self.aggre_weight_square,
            self.local_gradient_bound_square)  # p_i^2 * G_i^2
        f1 = matrix(np.eye(self.num_of_client) * f1_params)

        f2_params = BigA * np.multiply(
            self.local_staleness, self.local_gradient_bound)  # \tau_i * G_i
        f2 = matrix(-1 * np.eye(self.num_of_client) * f2_params)

        F = sparse([[f1, f2]])

        g = log(matrix(np.ones(2 * self.number_of_client)))

        K = [2 * self.number_of_client]
        G = None
        h = None

        A1 = matrix([[1.]])
        A = matrix([[1.]])
        for i in range(self.number_of_client - 1):
            A = sparse([[A], [A1]])

        b = matrix([1.])
        sol = solvers.gp(
            K, F, g, G, h, A, b,
            solver='mosek')['x']  # solve out the probabitliy of each client

        return sol
