"""
A personalized federated learning training session using FedRep.

Reference:

Collins et al., "Exploiting Shared Representations for Personalized Federated
Learning," in the Proceedings of ICML 2021.

https://arxiv.org/abs/2102.07078

Source code: https://github.com/lgcollins/FedRep
"""
import os
import sys

# Get the current directory of module1.py
pfl_bases = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(pfl_bases))

from bases import fedavg_personalized_server
from bases import fedavg_partial
from bases.client_callbacks import completion_callbacks
from bases.trainer_callbacks import semi_mixing_trainer_callbacks

import fedrep_client
import fedrep_trainer


def main():
    """
    A Plato personalized federated learning sesstion for FedBABU approach.
    """
    trainer = fedrep_trainer.Trainer
    client = fedrep_client.Client(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        callbacks=[
            completion_callbacks.ClientModelCompletionCallback,
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
