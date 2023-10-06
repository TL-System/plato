"""
The implementation of Ditto method based on the pFL framework of Plato.

Tian Li, et.al, Ditto: Fair and robust federated learning through personalization, 2021:
 https://proceedings.mlr.press/v139/li21h.html

Official code: https://github.com/litian96/ditto
Third-part code: https://github.com/lgcollins/FedRep

"""

from pflbases import fedavg_personalized_server
from pflbases import personalized_client
from pflbases import fedavg_partial
from pflbases.trainer_callbacks import mixing_trainer_callbacks

import ditto_trainer_callbacks
import ditto_trainer_v2 as ditto_trainer


def main():
    """
    A personalized federated learning sesstion for Ditto approach.
    """
    trainer = ditto_trainer.Trainer
    client = personalized_client.Client(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        trainer_callbacks=[
            mixing_trainer_callbacks.PersonalizedModelMetricCallback,
            ditto_trainer_callbacks.DittoStatusCallback,
        ],
    )
    server = fedavg_personalized_server.Server(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
