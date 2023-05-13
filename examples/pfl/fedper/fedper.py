"""
The implementation of FedPer method based on the plato's pFL code.

Manoj Ghuhan Arivazhagan, et.al, Federated learning with personalization layers, 2019.
https://arxiv.org/abs/1912.00818

Official code: None
Third-part code: https://github.com/jhoon-oh/FedBABU
"""

import os
import sys

# Add `bases` to the path
pfl_bases = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(pfl_bases))

from pflbases import fedavg_personalized_server
from pflbases import fedavg_partial
from pflbases.client_callbacks import personalized_completion_callbacks
from pflbases.trainer_callbacks import semi_mixing_trainer_callbacks

import fedper_client
import fedper_trainer


def main():
    """
    A Plato personalized federated learning sesstion for FedBABU approach.
    """
    trainer = fedper_trainer.Trainer
    client = fedper_client.Client(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        callbacks=[
            personalized_completion_callbacks.ClientModelPersonalizedCompletionCallback,
        ],
        trainer_callbacks=[
            semi_mixing_trainer_callbacks.PersonalizedModelMetricCallback,
            semi_mixing_trainer_callbacks.PersonalizedModelStatusCallback,
        ],
    )
    server = fedavg_personalized_server.Server(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
