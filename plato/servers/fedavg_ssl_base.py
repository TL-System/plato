"""
A federated learning server supporting the self-supervised learning.
"""

import random
import logging

from plato.servers import fedavg
from plato.config import Config
from plato.utils import fonts


class Server(fedavg.Server):
    """ Federated learning server to support the training of ssl models. """

    def choose_clients(self, clients_pool, clients_count):
        """ Choose a subset of the clients to participate in each round. """
        assert clients_count <= len(clients_pool)
        random.setstate(self.prng_state)

        # if we decide to perfrom the eval test in the final round
        if hasattr(
                Config().clients,
                "do_final_eval_test") and Config().clients.do_final_eval_test:

            # if the training reaches the final round
            if self.current_round == Config().trainer.rounds:
                # perfrom the eval test by mandotary
                Config().clients = Config().clients._replace(
                    eval_test_interval=1)
                Config().clients = Config().clients._replace(do_test=True)

                # select all clients
                clients_count = len(clients_pool)
                logging.info(
                    fonts.colourize(
                        f"\n Performing {Config().data.augment_transformer_name}'s linear evaluation on all clients at final round {self.current_round}/{Config().trainer.rounds}.",
                        colour='red',
                        style='bold'))

        # Select clients randomly
        selected_clients = random.sample(clients_pool, clients_count)

        self.prng_state = random.getstate()
        logging.info("[%s] Selected clients: %s", self, selected_clients)
        return selected_clients