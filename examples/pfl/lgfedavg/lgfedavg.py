"""
The implementation of LG-FedAvg method based on the plato's
pFL code.

Paul Pu Liang, et.al, Think Locally, Act Globally: Federated Learning with Local and Global Representations
https://arxiv.org/abs/2001.01523

Official code: https://github.com/pliang279/LG-FedAvg

"""
import os
import sys

# Get the current directory of module1.py
pfl_bases = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(pfl_bases))

from bases import fedavg_personalized_server
from bases import fedavg_partial
from bases.client_callbacks import completion_callbacks
from bases.trainer_callbacks import mixing_trainer_callbacks

import lgfedavg_client
import lgfedavg_trainer


def main():
    """
    A Plato personalized federated learning sesstion for LG-FedAvg approach.
    """
    trainer = lgfedavg_trainer.Trainer
    client = lgfedavg_client.Client(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        callbacks=[
            completion_callbacks.ClientModelCompletionCallback,
        ],
        trainer_callbacks=[
            mixing_trainer_callbacks.PersonalizedModelMetricCallback,
            mixing_trainer_callbacks.PersonalizedModelStatusCallback,
        ],
    )
    server = fedavg_personalized_server.Server(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
