"""
A customized server with asynchronous client selection
"""
import logging

from turtle import up, update
from plato.config import Config
from plato.servers import fedavg
from cvxopt import matrix, log, solvers, sparse
from collections import OrderedDict
from plato.config import Config
import mosek
import numpy as np
import torch


class Server(fedavg.Server):
    """A federated learning server."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        self.number_of_client = Config().clients.total_clients
        self.local_gradient_bounds = None
        self.aggregation_weights = None
        self.local_stalenesses = None
        self.squared_deltas_current_round = None
        self.alpha = None
        self.unexplored_clients = None

    def configure(self):
        """Initializes necessary variables."""
        super().configure()

        self.local_gradient_bounds = 0.5 * np.ones(
            self.number_of_client
        )  
        self.local_stalenesses = 0.01 * np.ones(self.number_of_client)
        self.aggregation_weights = np.ones(self.number_of_client) * (
            1.0 / self.number_of_client
        )
        self.unexplored_clients = list(range(self.number_of_client))
        self.alpha = 10

    def choose_clients(self, clients_pool, clients_count):
        """Choose a subset of the clients to participate in each round."""

        assert clients_count <= len(clients_pool)

        # Select clients based on calculated probability (within clients_pool)
        p = self.calculate_selection_probability(clients_pool)
        #logging.info(f"The calculated probability is: ", p)
        selected_clients = np.random.choice(
            clients_pool, clients_count, replace=False, p=p
        )

        logging.info("[%s] Selected clients: %s", self, selected_clients)
        return selected_clients.tolist()

    async def aggregate_deltas(self, updates, deltas_received):
        avg_update = await super().aggregate_deltas(updates, deltas_received)

        self.squared_deltas_current_round = np.zeros(self.number_of_client)
        sum_deltas_current_round = 0
        deltas_counter = 0
        
        # Find delta bound for each client
        for update, delta in zip(updates, deltas_received):
            squared_delta = 0
            for layer, value in delta.items():
                if "conv" in layer:
                    squared_value = np.sum(np.square(value.detach().cpu().numpy()))
                    squared_delta += squared_value
            self.squared_deltas_current_round[update.client_id - 1] = np.sqrt(
                squared_delta  # .item() if nlp task, comment out .item() as it's int object
            )
            # Remove explored client from unexplored list
            if (update.client_id - 1) in self.unexplored_clients:
                self.unexplored_clients.remove(update.client_id - 1)

            sum_deltas_current_round += np.sqrt(squared_delta)
            deltas_counter += 1

        # Calculate expect deltas for all unexplored clients based on client explored this round.
        avg_deltas_current_round = sum_deltas_current_round / deltas_counter
        expect_deltas = self.alpha * avg_deltas_current_round

        for client_counter in range(200):
            if client_counter in self.unexplored_clients:
                self.squared_deltas_current_round[client_counter] = expect_deltas
       
        return avg_update

    def weights_aggregated(self, updates):
        """ "Update clients record on the server"""

        # Extract the local staleness and update the record of client staleness.
        for update in updates:
            self.local_stalenesses[update.client_id - 1] = (
                update.staleness + 0.1
            )  

        # Update local gradient bounds
        for client_id, bound in enumerate(self.squared_deltas_current_round):
            if bound != 0:
                # Update local gradient bounds for all clients here
                self.local_gradient_bounds[client_id] = bound

        self.extract_aggregation_weights(updates)

    def extract_aggregation_weights(self, updates):
        """Extract aggregation weights"""
        # Extract the total number of samples
        self.total_samples = sum([update.report.num_samples for update in updates])

        for update in updates:
            self.aggregation_weights[update.client_id - 1] = (
                update.report.num_samples / self.total_samples
            )

    def calculate_selection_probability(self, clients_pool):
        """Calculte selection probability based on the formulated geometric optimization problem
        Minimize \beta \sum_{i=1}^N \frac{p_i^2 * G_i^2}{q_i} + A \sum_{i=1}^N q_i * \tau_i * G_i
        Subject to \sum_{i=1}{N} q_i = 1
                q_i > 0
        Probability Variables are q_i

        """
        logging.info("Calculating selection probabitliy ... ")
        beta = 1
        BigA = 1
        # Extract info for clients in the pool
        clients_pool = [
            index - 1 for index in clients_pool
        ]  # index in clients_pool starts from 1 rather than 0 in other np.arrays.
        num_of_clients_inpool = len(clients_pool)

        aggregation_weights_inpool = self.aggregation_weights[clients_pool]
        local_gradient_bounds_inpool = self.local_gradient_bounds[clients_pool]
        local_staleness_inpool = np.square(self.local_stalenesses[clients_pool])

        # Read aggre_weight
        aggre_weight_square = np.square(aggregation_weights_inpool)  # p_i^2
        local_gradient_bound_square = np.square(local_gradient_bounds_inpool)  # G_i^2

        f1_params = matrix(
            beta * np.multiply(aggre_weight_square, local_gradient_bound_square)
        )  # p_i^2 * G_i^2

        # f1 = matrix(np.eye(num_of_clients_inpool) * f1_params)
        f2_temp = np.multiply(local_staleness_inpool, local_gradient_bounds_inpool)
        f2_params = matrix(
            BigA * np.multiply(aggre_weight_square, f2_temp)
        )  # p_i^2 * \tau_i * G_i 


        f1 = matrix(-1 * np.eye(num_of_clients_inpool))
        f2 = matrix(np.eye(num_of_clients_inpool))
        F = sparse([[f1, f2]])

        g = log(matrix(sparse([[f1_params, f2_params]])))

        K = [2 * num_of_clients_inpool]
        G = matrix(-1 * np.eye(num_of_clients_inpool))  
        h = matrix(np.zeros((num_of_clients_inpool, 1)))  

        A1 = matrix([[1.0]])
        A = matrix([[1.0]])
        for i in range(num_of_clients_inpool - 1):
            A = sparse([[A], [A1]])

        b = matrix([1.0])
        solvers.options["maxiters"] = 500
        sol = solvers.gp(K, F, g, G, h, A, b, solver="mosek")[
            "x"
        ]  # solve out the optimal sampling probabitliy

        return np.array(sol).reshape(-1)

