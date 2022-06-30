"""
A customized server with asynchronous client selection
"""
from plato.config import Config
from plato.servers import fedavg
from cvxopt import matrix, log, solvers,sparse
import mosek
import numpy as np

class Server(fedavg.Server):
    """A federated learning server."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)

    def choose_clients(self, clients_pool, clients_count):
        return super().choose_clients(clients_pool, clients_count)
    
    def calculate_selection_probability(self, aggre_weight, local_gradient_bound, local_staleness)
    """Calculte selection probability based on the formulated geometric optimization problem
        Minimize \alpha \sum_{i=1}^N \frac{p_i^2 * G_i^2}{q_i} + A \sum_{i=1}^N q_i * \tau_i * G_i
        Subject to \sum_{i=1}{N} q_i = 1
                   q_i > 0 
        Probability Variables are q_i

    """

    aggre_weight_square = np.square(aggre_weight)  # p_i^2
    local_gradient_bound_square = np.square(local_gradient_bound)  # G_i^2

    F1_params = alpha * np.multiply(aggre_weight_square,
                            local_gradient_bound_square)  # p_i^2 * G_i^2
    F1 = matrix(np.eye(num) * F1_params)

    F2_params = BigA * np.multiply(local_staleness, local_gradient_bound) # \tau_i * G_i
    F2 = matrix(-1 * np.eye(num) * F2_params )

    F = sparse([[F1,F2]])

    g = log( matrix( np.ones(2 * numberOfClient)))
    
    K = [2 * numberOfClient]
    G = None
    h = None

    A1 = matrix([[1.]])
    A = matrix([[1.]])
    for i in range(numberOfClient - 1):
        A = sparse([[A], [A1]])

    b = matrix([1.])
    sol = solvers.gp(K, F, g, G, h, A, b,solver='mosek')['x'] # solve out the probabitliy of each client

    return sol

