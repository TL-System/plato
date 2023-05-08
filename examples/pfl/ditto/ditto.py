"""
The implementation of Ditto method based on the pFL framework of Plato.

Tian Li, et.al, Ditto: Fair and robust federated learning through personalization, 2021:
 https://proceedings.mlr.press/v139/li21h.html

Official code: https://github.com/litian96/ditto
Third-part code: https://github.com/lgcollins/FedRep

"""

import os
import sys

# Get the current directory of module1.py
pfl_bases = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(pfl_bases))

from bases import fedavg_personalized_server
from bases import fedavg_partial

from bases.trainer_callbacks import (
    PersonalizedLogMetricCallback,
    PersonalizedLogProgressCallback,
)

import ditto_trainer_callbacks
import ditto_client
import ditto_trainer_v2 as ditto_trainer


def main():
    """
    A Plato personalized federated learning sesstion for FedBABU approach.
    """
    trainer = ditto_trainer.Trainer
    client = ditto_client.Client(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        trainer_callbacks=[
            PersonalizedLogMetricCallback,
            PersonalizedLogProgressCallback,
            ditto_trainer_callbacks.PersonalizedLogModelCallback,
            ditto_trainer_callbacks.PersonalizedModelMetricCallback,
        ],
    )
    server = fedavg_personalized_server.Server(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
