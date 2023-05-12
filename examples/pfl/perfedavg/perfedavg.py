"""
The implementation of Per-FedAvg method based on the plato's
pFL code.

Alireza Fallah, et.al, Personalized federated learning with theoretical guarantees:
A model-agnostic meta-learning approach, NeurIPS 2020.
https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html

Official code: None
Third-part code: https://github.com/jhoon-oh/FedBABU

"""
import os
import sys

# Get the current directory of module1.py
pfl_bases = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(pfl_bases))

from bases import fedavg_personalized_server
from bases import fedavg_partial
from bases import personalized_client
from bases.trainer_callbacks import separate_trainer_callbacks
from bases.client_callbacks import completion_callbacks

import perfedavg_trainer


def main():
    """
    A Plato personalized federated learning sesstion for FedBABU approach.
    """
    trainer = perfedavg_trainer.Trainer
    client = personalized_client.Client(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        callbacks=[
            completion_callbacks.ClientModelCompletionCallback,
        ],
        trainer_callbacks=[
            separate_trainer_callbacks.PersonalizedModelMetricCallback,
            separate_trainer_callbacks.PersonalizedModelStatusCallback,
        ],
    )
    server = fedavg_personalized_server.Server(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
