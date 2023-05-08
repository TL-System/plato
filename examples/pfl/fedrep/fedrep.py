"""
A personalized federated learning training session using FedRep.

Reference:

Collins et al., "Exploiting Shared Representations for Personalized Federated
Learning," in the Proceedings of ICML 2021.

https://arxiv.org/abs/2102.07078

Source code: https://github.com/lgcollins/FedRep
"""

import fedrep_trainer
import fedrep_client

from examples.pfl.bases.fedavg_personalized import Server


def main():
    """
    A Plato federated learning training session using the FedRep algorithm under the
    supervised learning setting.
    """
    trainer = fedrep_trainer.Trainer
    client = fedrep_client.Client(trainer=trainer)
    server = Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
