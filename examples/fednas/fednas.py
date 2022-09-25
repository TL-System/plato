"""
A personalized federated learning training session using FedRep.

Reference:

Collins et al., "Exploiting Shared Representations for Personalized Federated
Learning", in the Proceedings of ICML 2021.

https://arxiv.org/abs/2102.07078

Source code: https://github.com/lgcollins/FedRep
"""

import fednas_client
import fednas_server
from plato.trainers.basic import Trainer
import fednas_algorithm
from Darts.model_search import Network
from Darts.architect import Architect

def main():
    """ A Plato federated learning training session using the FedNAS algorithm. """
    client = fednas_client.Client(model=Network,algorithm=fednas_algorithm.ClientAlgorithm, trainer=Trainer)
    server = fednas_server.Server(model=Architect,algorithm=fednas_algorithm.ServerAlgorithm, trainer=Trainer)

    server.run(client)


if __name__ == "__main__":
    main()
