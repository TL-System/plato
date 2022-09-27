"""
An asynchronous federated learning server using Sirius.
"""

import math
import random
import logging
import asyncio
from plato.config import Config
from plato.servers import fedavg

from inspect import currentframe

def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno

class Server(fedavg.Server):
    """A federated learning server using the sirius algorithm."""

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        self.staleness_factor = 0.5
        self.exploration_factor = Config().server.exploration_factor
        self.explored_clients = []
        self.unexplored_clients = []
        print("Server finished line_", get_linenumber())

    def configure(self):
        """Initialize necessary variables."""
        super().configure()

        self.client_utilities = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.unexplored_clients = list(range(1, self.total_clients + 1))
        print("Server finished line_", get_linenumber())

    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using federated averaging with calcuated staleness factor."""
        print("Server finished line_", get_linenumber()) 
        
        # Extract the total number of samples
        self.total_samples = sum(update.report.num_samples for update in updates)

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        for i, update in enumerate(deltas_received):
            report = updates[i].report
            num_samples = report.num_samples
            staleness = updates[i].staleness
            staleness_factor = self.staleness_function(staleness)

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples) * staleness_factor # no normalization but from Zhifeng's code.

            # Yield to other tasks in the server
            await asyncio.sleep(0)
        print("Server finished line_", get_linenumber())
        return avg_update

    def staleness_function(self, stalenss):
        print("Server finished line_", get_linenumber())
        return 1.0 / pow(stalenss + 1, self.staleness_factor) # formula obtained from Zhifeng's code. (clients_manager/sirius/staleness_factor_calculator)

    def weights_aggregated(self, updates):
        """Method called at the end of aggregating received weights."""
        """Calculate client utility here and update the record on the server"""
        for update in updates:
            self.client_utilities[update.client_id] = update.report.statistics_utility * self.staleness_function(update.staleness)
        print("Server finished line_", get_linenumber())

    def choose_clients(self, clients_pool, clients_count):
        """Choose a subset of the clients to participate in each round."""
        selected_clients = []
        print("Server finished line_", get_linenumber())
        if self.current_round > 1:
            # Exploitation
            exploit_client_num = max(
                math.ceil((1.0 - self.exploration_factor) * clients_count),
                clients_count - len(self.unexplored_clients),
            )

            sorted_util = sorted(
                self.client_utilities, key=self.client_utilities.get, reverse=True
            )
            selected_clients = sorted_util[:exploit_client_num]
        print("Server finished line_", get_linenumber())
        # Exploration
        random.setstate(self.prng_state)

        # Select unexplored clients randomly
        selected_unexplore_clients = random.sample(
            self.unexplored_clients, clients_count - len(selected_clients)
        )

        self.prng_state = random.getstate()
        self.explored_clients += selected_unexplore_clients

        for client_id in selected_unexplore_clients:
            self.unexplored_clients.remove(client_id)
        print("Server finished line_", get_linenumber())
        selected_clients += selected_unexplore_clients

        #for client in selected_clients:
        #    self.client_selected_times[client] += 1

        logging.info("[%s] Selected clients: %s", self, selected_clients)

        return selected_clients
           
