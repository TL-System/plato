"""
An implementation of the FedBABU algorithm.

J. Oh, et al., "FedBABU: Toward Enhanced Representation for Federated Image Classification,"
in the Proceedings of ICLR 2022.

https://openreview.net/pdf?id=HuaYQfggn5u

Source code: https://github.com/jhoon-oh/FedBABU
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

import fedbabu_client_callbacks
import fedbabu_client
import fedbabu_trainer


def main():
    """
    A Plato personalized federated learning sesstion for FedBABU approach.
    """
    trainer = fedbabu_trainer.Trainer
    client = fedbabu_client.Client(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        callbacks=[
            fedbabu_client_callbacks.ClientModelCallback,
        ],
        trainer_callbacks=[
            PersonalizedLogMetricCallback,
            PersonalizedLogProgressCallback,
        ],
    )
    server = fedavg_personalized_server.Server(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
