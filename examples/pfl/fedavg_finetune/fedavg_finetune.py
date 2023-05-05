"""
An implementation of the personalized learning variant of FedAvg.

Such an variant of FedAvg is recently mentioned and discussed in work [1].

[1] Liam Collins, et al., "Exploiting shared representations for personalized federated learning,"
in the Proceedings of ICML 2021.

    Address: https://proceedings.mlr.press/v139/collins21a.html

    Code: https://github.com/lgcollins/FedRep

"""

import os
import sys

# Get the current directory of module1.py
pfl_bases = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(pfl_bases))


from bases.personalized_trainer import Trainer
from bases.fedavg_personalized_server import Server

from bases.client_callbacks import ClientModelCallback
from bases.trainer_callbacks import TrainerPersonalizationCallback


import fedavg_finetune_client


def main():
    """
    A Plato personalized federated learning training session using the FedAvg algorithm under the
    supervised learning setting.
    """
    trainer = Trainer
    client = fedavg_finetune_client.Client(
        trainer=trainer,
        callbacks=[ClientModelCallback],
        trainer_callbacks=[TrainerPersonalizationCallback],
    )
    server = Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
