"""
A personalized federated learning training session using FedRep.

Reference:

Collins et al., "Exploiting Shared Representations for Personalized Federated
Learning," in the Proceedings of ICML 2021.

https://arxiv.org/abs/2102.07078

Source code: https://github.com/lgcollins/FedRep
"""

import fedrep_trainer

from plato.clients import fedavg_personalized as personalized_client
from plato.servers import fedavg_personalized as personalized_server


def main():
    """
    A personalized federated learning session using FedRep.
    """
    trainer = fedrep_trainer.Trainer
    client = personalized_client.Client(trainer=trainer)
    server = personalized_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
