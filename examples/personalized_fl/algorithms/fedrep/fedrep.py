"""
A personalized federated learning training session using FedRep.

Reference:

Collins et al., "Exploiting Shared Representations for Personalized Federated
Learning," in the Proceedings of ICML 2021.

https://arxiv.org/abs/2102.07078

Source code: https://github.com/lgcollins/FedRep
"""

from pflbases import fedavg_personalized
from pflbases import fedavg_partial
from pflbases import fedavg_personalized_client
import fedrep_trainer


def main():
    """
    A personalized federated learning session for FedRep approach.
    """
    trainer = fedrep_trainer.Trainer
    client = fedavg_personalized_client.Client(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )
    server = fedavg_personalized.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
