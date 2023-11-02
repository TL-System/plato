"""
The implementation of Ditto method based on the pFL framework of Plato.

Reference:
Tian Li, et al., Ditto: Fair and robust federated learning through personalization, 2021:
 https://proceedings.mlr.press/v139/li21h.html

Official code: https://github.com/litian96/ditto
Third-party code: https://github.com/lgcollins/FedRep

"""

from pflbases import fedavg_personalized
from pflbases import fedavg_personalized
from pflbases import fedavg_personalized

import ditto_trainer


def main():
    """
    A personalized federated learning session for Ditto approach.
    """
    trainer = ditto_trainer.Trainer
    client = fedavg_personalized.Client(
        trainer=trainer,
        algorithm=fedavg_personalized.Algorithm,
    )
    server = fedavg_personalized.Server(
        trainer=trainer,
        algorithm=fedavg_personalized.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
