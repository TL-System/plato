"""
A personalized federated learning training session using FedRep.

Reference:

Collins et al., "Exploiting Shared Representations for Personalized Federated
Learning", in the Proceedings of ICML 2021.

https://arxiv.org/abs/2102.07078

Source code: https://github.com/lgcollins/FedRep
"""

import fedrep_trainer
import fedrep_client
import fedrep_server
import fedrep_algorithm


def main():
    """ A Plato federated learning training session using the FedRep algorithm. """
    trainer = fedrep_trainer.Trainer
    algorithm = fedrep_algorithm.Algorithm
    client = fedrep_client.Client(algorithm=algorithm, trainer=trainer)
    server = fedrep_server.Server(algorithm=algorithm, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
